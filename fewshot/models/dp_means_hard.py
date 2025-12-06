import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 

from fewshot.models.model_factory import RegisterModel
from fewshot.models.imp import IMPModel
from fewshot.models.utils import *
from fewshot.models.weighted_ce_loss import weighted_loss
import pdb

@RegisterModel("dp-means-hard")
class DPMeansHardModel(IMPModel):

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)
        #initialize labels
        y_train_np = batch.y_train.detach().cpu().numpy()
        nClusters = len(np.unique(y_train_np))
        nInitialClusters = nClusters

        #get embedded representations
        h_train = self._run_forward(batch.x_train)
        h_test = self._run_forward(batch.x_test)

        prob_train = one_hot(batch.y_train, nClusters).to(DEVICE)

        if batch.x_unlabel is None:
            protos_clusters = [self._compute_protos(h_train, prob_train)]

        #this is unused, but maximizes function sharing
        bsize = h_train.size()[0]
        radii = torch.ones(bsize, nClusters).to(DEVICE) * torch.exp(self.log_sigma_l)

        #initialize prototypes and target labels
        target_labels = torch.arange(0, nClusters).long()
        protos = self._compute_protos(h_train[:,:,:], prob_train[:,:,:])


        #estimate lamda
        lamda = self.estimate_lambda(protos.detach(), batch.x_unlabel is not None)


        #initialize assignments
        if batch.y_unlabel is not None:
            assignments = torch.arange(0, batch.y_train.size()[1] + batch.y_unlabel.size()[1]).long()
        else:
            assignments = batch.y_train[0].detach().cpu().long()

        for ii in range(self.config.num_cluster_steps):
            tensor_proto = protos.detach()
            #get assignments for training examples
            for i, ex in enumerate(h_train[0]):
                    idxs = torch.nonzero(target_labels == y_train_np[0,i])[0]
                    distances = self._compute_distances(tensor_proto[:,idxs.numpy(),:], ex.detach())
                    if (torch.min(distances) > lamda):
                        nClusters, tensor_proto, radii  = self._add_cluster(nClusters, tensor_proto, radii, cluster_type='labeled', ex=ex.detach())
                        target_labels = torch.cat([target_labels, batch.y_train[0, i].detach().cpu().long().unsqueeze(0)], dim=0)
                        assignments[i] = nClusters - 1
                    else:
                        label = np.argmin(distances)
                        assignments[i] = idxs[label]

            nTrainClusters = nClusters
            if batch.x_unlabel is not None:
                h_unlabel = self._run_forward(batch.x_unlabel.to(DEVICE))
                h_all = torch.cat([h_train, h_unlabel], dim=1)
                #get assignments for unlabeled examples (hard assignments)
                for i, ex in enumerate(h_unlabel[0]):
                        distances = self._compute_distances(tensor_proto, ex.detach())
                        if torch.min(distances) > lamda:
                            nClusters, tensor_proto, radii = self._add_cluster(nClusters, tensor_proto, radii, cluster_type='unlabeled', ex=ex.detach())
                            target_labels = torch.cat([target_labels, torch.LongTensor([-1])],dim=0)
                            assignments[i+batch.y_train.size()[1]] = nClusters - 1
                        else:
                            assignments[i+batch.y_train.size()[1]] = np.argmin(distances)

            else:
                h_all = h_train

        final_assigns = one_hot(assignments.unsqueeze(0), nClusters).to(DEVICE)
        final_protos = self._compute_protos(h_all, final_assigns)

        protos, radii, target_labels = self.delete_empty_clusters(final_protos, final_assigns, radii, target_labels.to(DEVICE))
        
        logits = compute_logits(protos, h_test).squeeze()
        
        # convert class targets into indicators for supports in each class
        labels = batch.y_test
        labels = torch.where(labels >= nInitialClusters, torch.tensor(-1, device=labels.device, dtype=labels.dtype), labels)

        support_targets = labels[0, :, None] == target_labels
        loss = self.loss(logits, support_targets, target_labels)

        # map support predictions back into classes to check accuracy
        _, support_preds = torch.max(logits.detach(), dim=1)
        y_pred = target_labels[support_preds]

        # Handle case where test set is empty (can happen in some episodes)
        if labels.size(1) == 0:
            acc_val = 0.0
        else:
            acc_val = torch.eq(y_pred, labels[0]).float().mean().item()
        return loss, {
            'loss': loss.item(),
            'acc': acc_val,
            'logits': logits.detach().cpu().numpy()
            }
