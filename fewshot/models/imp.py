import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fewshot.models.model_factory import RegisterModel
from fewshot.models.basic import Protonet
from fewshot.models.utils import *
from fewshot.models.weighted_ce_loss import weighted_loss
import pdb

@RegisterModel("imp")
class IMPModel(Protonet):

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled', ex=None):
        nClusters += 1
        bsize = protos.size(0)
        d_radii = torch.ones(bsize, 1, device=protos.device)

        if cluster_type == 'labeled':
            d_radii = d_radii * torch.exp(self.log_sigma_l)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_u)

        if ex is None:
            new_proto = self.base_distribution.data.to(protos.device)
        else:
            new_proto = ex.unsqueeze(0).unsqueeze(0).to(protos.device)

        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def estimate_lambda(self, tensor_proto, semi_supervised):
        rho = tensor_proto[0].var(dim=0).mean()

        if semi_supervised:
            sigma = (torch.exp(self.log_sigma_l).item() + torch.exp(self.log_sigma_u).item()) / 2.
        else:
            sigma = torch.exp(self.log_sigma_l).item()

        lamda = -2 * sigma * np.log(self.config.ALPHA) + self.config.dim * sigma * np.log(1 + rho.item() / sigma)
        return lamda

    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = prob[0].sum(dim=0)
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs]

    def loss(self, logits, targets, labels):
        targets = targets.to(logits.device)
        target_logits = torch.full_like(logits, float('-Inf'))
        target_logits[targets] = logits[targets]
        _, best_targets = torch.max(target_logits, dim=1)

        weights = torch.zeros_like(logits)
        unique_labels = torch.unique(labels)
        for l in unique_labels:
            class_mask = labels == l
            class_logits = torch.full_like(logits, float('-Inf'))
            class_logits[:, class_mask] = logits[:, class_mask]
            _, best_in_class = torch.max(class_logits, dim=1)
            weights[torch.arange(logits.size(0)), best_in_class] = 1.

        return weighted_loss(logits, best_targets, weights).mean()

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)
        nClusters = len(torch.unique(batch.y_train))
        nInitialClusters = nClusters

        h_train = self._run_forward(batch.x_train)
        h_test = self._run_forward(batch.x_test)

        prob_train = one_hot(batch.y_train, nClusters).to(h_train.device)

        bsize = h_train.size(0)
        radii = torch.ones(bsize, nClusters, device=h_train.device) * torch.exp(self.log_sigma_l)
        support_labels = torch.arange(0, nClusters, device=h_train.device).long()

        protos = self._compute_protos(h_train, prob_train)
        lamda = self.estimate_lambda(protos.data, batch.x_unlabel is not None)

        for _ in range(self.config.num_cluster_steps):
            tensor_proto = protos.data

            # Labeled examples
            for i, ex in enumerate(h_train[0]):
                idxs = torch.nonzero(batch.y_train[0, i] == support_labels)[0]
                distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)
                if torch.min(distances) > lamda:
                    nClusters, tensor_proto, radii = self._add_cluster(
                        nClusters, tensor_proto, radii, cluster_type='labeled', ex=ex.data
                    )
                    labeled_flag = batch.y_train[0, i].unsqueeze(0)
                    support_labels = torch.cat([support_labels, labeled_flag], dim=0)

            if nClusters > nInitialClusters:
                support_targets = batch.y_train[0, :, None] == support_labels
                prob_train = assign_cluster_radii_limited(tensor_proto, h_train, radii, support_targets)

            nTrainClusters = nClusters

            # Unlabeled examples
            if batch.x_unlabel is not None:
                h_unlabel = self._run_forward(batch.x_unlabel)
                h_all = torch.cat([h_train, h_unlabel], dim=1)
                unlabeled_flag = torch.tensor([-1], device=h_train.device)

                for i, ex in enumerate(h_unlabel[0]):
                    distances = self._compute_distances(tensor_proto, ex.data)
                    if torch.min(distances) > lamda:
                        nClusters, tensor_proto, radii = self._add_cluster(
                            nClusters, tensor_proto, radii, cluster_type='unlabeled', ex=ex.data
                        )
                        support_labels = torch.cat([support_labels, unlabeled_flag], dim=0)

                if nClusters > nTrainClusters:
                    unlabeled_clusters = torch.zeros(prob_train.size(0), prob_train.size(1), nClusters - nTrainClusters, device=h_train.device)
                    prob_train = torch.cat([prob_train, unlabeled_clusters], dim=2)

                prob_unlabel = assign_cluster_radii(tensor_proto, h_unlabel, radii)
                prob_unlabel_nograd = prob_unlabel.detach()
                prob_all = torch.cat([prob_train.detach(), prob_unlabel_nograd], dim=1)

                protos = self._compute_protos(h_all, prob_all)
                protos, radii, support_labels = self.delete_empty_clusters(protos, prob_all, radii, support_labels)
            else:
                protos = tensor_proto
                protos = self._compute_protos(h_train, prob_train.detach())
                protos, radii, support_labels = self.delete_empty_clusters(protos, prob_train, radii, support_labels)

        logits = compute_logits_radii(protos, h_test, radii).squeeze()
        labels = batch.y_test
        labels[labels >= nInitialClusters] = -1
        support_targets = labels[0, :, None] == support_labels
        loss = self.loss(logits, support_targets, support_labels)

        _, support_preds = torch.max(logits, dim=1)
        y_pred = support_labels[support_preds]
        acc_val = torch.eq(y_pred, labels[0]).float().mean()

        return loss, {'loss': loss.item(), 'acc': acc_val, 'logits': logits[0].detach()}

    def forward_unsupervised(self, sample, super_classes, unlabel_lambda=20., num_cluster_steps=5):
        batch = self._process_batch(sample, super_classes=super_classes)
        h_test = self._run_forward(batch.x_test)

        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = h_unlabel
            protos = h_unlabel[0][0].unsqueeze(0).unsqueeze(0)
            radii = torch.ones(1, 1, device=h_unlabel.device) * torch.exp(self.log_sigma_l)
            nClusters = 1

            for _ in range(num_cluster_steps):
                tensor_proto = protos.data
                for i, ex in enumerate(h_unlabel[0]):
                    distances = self._compute_distances(tensor_proto, ex.data)
                    if torch.min(distances) > unlabel_lambda:
                        nClusters, tensor_proto, radii = self._add_cluster(
                            nClusters, tensor_proto, radii, cluster_type='labeled', ex=ex.data
                        )

                prob_unlabel = assign_cluster_radii(tensor_proto, h_unlabel, radii)
                prob_unlabel_nograd = prob_unlabel.detach()
                protos = self._compute_protos(h_all, prob_unlabel_nograd)

        return {'logits': prob_unlabel_nograd[0].detach()}
