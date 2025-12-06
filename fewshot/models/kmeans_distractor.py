
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 

from fewshot.models.model_factory import RegisterModel
from fewshot.models.imp import IMPModel
from fewshot.models.utils import *
from fewshot.models.weighted_ce_loss import weighted_loss

@RegisterModel("kmeans-distractor")
class KMeansDistractorModel(IMPModel):

    def __init__(self, config, data):
        super(KMeansDistractorModel, self).__init__(config, data)

        self.base_distribution = (0*torch.randn(1, 1, config.dim)).to(DEVICE)

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled'):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius
            cluster_type: labeled or unlabeled
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        new_proto = self.base_distribution.clone().detach().to(DEVICE)

        protos = torch.cat([protos, new_proto], dim=1)
        zero_count = torch.zeros(bsize, 1).to(DEVICE)

        d_radii = torch.ones(bsize, 1).to(DEVICE).requires_grad_(True)

        if cluster_type == 'unlabeled':
            d_radii = d_radii * torch.exp(self.log_sigma_u)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_l)

        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def delete_empty_clusters(self, tensor_proto, prob, radii, eps=1e-6):
        column_sums = torch.sum(prob[0],dim=0).detach()
        good_protos = column_sums > eps
        idxs = good_protos.nonzero(as_tuple=False).view(-1)
        return tensor_proto[:,idxs,:], radii[:,idxs]

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)

        nClusters = len(np.unique(batch.y_train.detach().cpu().numpy()))
        nClustersInitial = nClusters

        h_train = self._run_forward(batch.x_train)
        h_test = self._run_forward(batch.x_test)

        prob_train = one_hot(batch.y_train, nClusters).to(DEVICE)

        protos = self._compute_protos(h_train, prob_train)

        bsize = h_train.size()[0]

        radii = torch.exp(self.log_sigma_l) * torch.ones(bsize, nClusters).to(DEVICE).detach()

        support_labels = torch.arange(0, nClusters).to(DEVICE).long()
        unlabeled_flag = torch.LongTensor([-1]).to(DEVICE)

        #deal with semi-supervised data
        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = torch.cat([h_train, h_unlabel], dim=1)

            #add in distractor cluster centered at zero
            nClusters, protos, radii = self._add_cluster(nClusters, protos, radii, 'unlabeled')
            prob_train = one_hot(batch.y_train, nClusters).to(DEVICE)
            support_labels = torch.cat([support_labels, unlabeled_flag], dim=0)

            #perform some clustering steps
            for ii in range(self.config.num_cluster_steps):

                prob_unlabel = assign_cluster_radii(protos, h_unlabel, radii)
                prob_unlabel_nograd = prob_unlabel.detach().to(DEVICE)

                prob_all = torch.cat([prob_train, prob_unlabel_nograd], dim=1)
                protos = self._compute_protos(h_all, prob_all)


        logits = compute_logits_radii(protos, h_test, radii).squeeze()

        labels = batch.y_test
        labels = torch.where(labels >= nClustersInitial, torch.tensor(-1, device=labels.device, dtype=labels.dtype), labels)

        support_targets = labels[0, :, None] == support_labels
        loss = self.loss(logits, support_targets, support_labels)

        # map support predictions back into classes to check accuracy
        _, support_preds = torch.max(logits.detach(), dim=1)
        y_pred = support_labels[support_preds]

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
