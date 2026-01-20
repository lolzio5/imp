import torch
from typing import Optional, Tuple, Dict, Any
from fewshot.models.model_factory import RegisterModel
from fewshot.models.basic import Protonet
from fewshot.models.utils import *
from fewshot.models.weighted_ce_loss import weighted_loss


@RegisterModel("imp")
class IMPModel(Protonet):
    def _add_cluster(self, nClusters: int, protos: torch.Tensor, radii: torch.Tensor, 
        cluster_type: str = 'unlabeled',
        ex: Optional[torch.Tensor] = None
    ):
        nClusters += 1
        bsize = protos.size(0)
        
        d_radii = torch.ones(bsize, 1, device=protos.device, dtype=protos.dtype)
        
        if cluster_type == 'labeled':
            d_radii = d_radii * torch.exp(self.log_sigma_l)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_u)
        
        new_proto = ex.unsqueeze(0).unsqueeze(0)
        
        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        
        return nClusters, protos, radii

    def estimate_lambda(self, tensor_proto: torch.Tensor, semi_supervised: bool):
        rho = tensor_proto[0].var(dim=0).mean()
        
        if semi_supervised:
            sigma = (torch.exp(self.log_sigma_l) + torch.exp(self.log_sigma_u)) / 2.0
        else:
            sigma = torch.exp(self.log_sigma_l)
        
        lamda = 2 * sigma * torch.log(torch.tensor(self.config.ALPHA, device=sigma.device)) - \
                self.config.dim * sigma * torch.log(1 + rho / sigma)
        
        return lamda
    
    def delete_empty_clusters(self, tensor_proto: torch.Tensor, prob: torch.Tensor, radii: torch.Tensor, 
        targets: torch.Tensor,
        eps: float = 1e-3 ):
        column_sums = torch.sum(prob[0], dim=0)
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos, as_tuple=False).squeeze()
        
        if idxs.dim() == 0:
            idxs = idxs.unsqueeze(0)
        
        return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs]
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor):
        targets = targets.to(logits.device)
        
        target_logits = torch.full_like(logits, float('-inf'))
        target_logits[targets] = logits[targets]
        _, best_targets = torch.max(target_logits, dim=1)
        
        weights = torch.zeros_like(logits)
        
        unique_labels = torch.unique(labels)
        
        for l in unique_labels:
            class_mask = (labels == l)
            class_logits = torch.full_like(logits, float('-inf'))
            class_logits[class_mask.repeat(logits.size(0), 1)] = logits[:, class_mask].reshape(-1)
            _, best_in_class = torch.max(class_logits, dim=1)
            weights[torch.arange(targets.size(0), device=weights.device), best_in_class] = 1.0
        loss = weighted_loss(logits, best_targets, weights)
        return loss.mean()

    def forward(self,sample: Any, super_classes: bool = False):
        
        batch = self._process_batch(sample, super_classes=super_classes)
        unique_train_labels = torch.unique(batch.y_train)
        nClusters = unique_train_labels.numel()
        nInitialClusters = nClusters
        
        h_train = self._run_forward(batch.x_train)
        h_test = self._run_forward(batch.x_test)
        
        prob_train = one_hot(batch.y_train, nClusters).to(h_train.device)
        
        bsize = h_train.size(0)
        radii = torch.ones(bsize, nClusters, device=h_train.device) * torch.exp(self.log_sigma_l)
        
        support_labels = torch.arange(0, nClusters, device=h_train.device, dtype=torch.long)
        
        protos = self._compute_protos(h_train, prob_train)
        
        lamda = self.estimate_lambda(protos, batch.x_unlabel is not None)
        
        for ii in range(self.config.num_cluster_steps):
            tensor_proto = protos
            
            for i, ex in enumerate(h_train[0]):
                idxs = torch.nonzero(batch.y_train[0, i] == support_labels, as_tuple=False)[0]
            
                distances = self._compute_distances(tensor_proto[:, idxs, :], ex)
                
                labeled_flag = batch.y_train[0, i].unsqueeze(0)
                
                if torch.min(distances) > lamda:
                    nClusters, tensor_proto, radii = self._add_cluster(
                        nClusters, tensor_proto, radii, 
                        cluster_type='labeled', ex=ex
                    )
                    support_labels = torch.cat([support_labels, labeled_flag], dim=0)
            
            if nClusters > nInitialClusters:
                support_targets = batch.y_train[0, :, None] == support_labels
                prob_train = assign_cluster_radii_limited(
                    tensor_proto, h_train, radii, support_targets
                )
            
            nTrainClusters = nClusters
            
            if batch.x_unlabel is not None:
                h_unlabel = self._run_forward(batch.x_unlabel)
                h_all = torch.cat([h_train, h_unlabel], dim=1)
                unlabeled_flag = torch.tensor([-1], device=h_train.device, dtype=torch.long)
                
                for i, ex in enumerate(h_unlabel[0]):
                    distances = self._compute_distances(tensor_proto, ex)
                    if torch.min(distances) > lamda:
                        nClusters, tensor_proto, radii = self._add_cluster(
                            nClusters, tensor_proto, radii, 
                            cluster_type='unlabeled', ex=ex
                        )
                        support_labels = torch.cat([support_labels, unlabeled_flag], dim=0)
                
                if nClusters > nTrainClusters:
                    unlabeled_clusters = torch.zeros(
                        prob_train.size(0), prob_train.size(1), 
                        nClusters - nTrainClusters,
                        device=prob_train.device,
                        dtype=prob_train.dtype
                    )
                    prob_train = torch.cat([prob_train, unlabeled_clusters], dim=2)
                
                prob_unlabel = assign_cluster_radii(tensor_proto, h_unlabel, radii)
                prob_unlabel_nograd = prob_unlabel.detach()
                
                prob_all = torch.cat([prob_train, prob_unlabel_nograd], dim=1)
                protos = self._compute_protos(h_all, prob_all)
                
                protos, radii, support_labels = self.delete_empty_clusters(
                    protos, prob_all, radii, support_labels
                )
            else:
                protos = self._compute_protos(h_train, prob_train.detach())
                
                protos, radii, support_labels = self.delete_empty_clusters(
                    protos, prob_train, radii, support_labels
                )
        
        logits = compute_logits_radii(protos, h_test, radii).squeeze()
        
        labels = batch.y_test.clone()
        labels[labels >= nInitialClusters] = -1
        
        support_targets = labels[0, :, None] == support_labels
        
        loss = self.loss(logits, support_targets, support_labels)
        
        _, support_preds = torch.max(logits, dim=1)
        
        y_pred = support_labels[support_preds]
        acc_val = torch.eq(y_pred, labels[0]).float().mean()
        num_prototypes = protos.size(1)
        return loss, {
            'loss': loss,
            'acc': acc_val,
            'logits': logits,
            'num_protos':num_prototypes
        }