# import torch
# from typing import Optional, Tuple, Dict, Any
# from fewshot.models.model_factory import RegisterModel
# from fewshot.models.basic import Protonet
# from fewshot.models.utils import *
# from fewshot.models.weighted_ce_loss import weighted_loss


# @RegisterModel("imp")
# class IMPModel(Protonet):
#     def _add_cluster(
#         self,
#         nClusters: int,
#         protos: torch.Tensor,
#         radii: torch.Tensor,
#         cluster_type: str = "unlabeled",
#         ex: Optional[torch.Tensor] = None
#     ) -> Tuple[int, torch.Tensor, torch.Tensor]:
#         """
#         Add a new cluster whose mean is `ex`
#         """
#         assert ex is not None

#         protos = torch.cat([protos, ex.unsqueeze(0)], dim=0)

#         if cluster_type == "labeled":
#             r = self.log_sigma_l
#         else:
#             r = self.log_sigma_u

#         radii = torch.cat([radii, r.view(1)], dim=0)

#         return nClusters + 1, protos, radii


#     def estimate_lambda(
#         self,
#         tensor_proto: torch.Tensor,
#         semi_supervised: bool
#     ) -> torch.Tensor:
#         """
#         Episodic λ estimation (Eq. 5)
#         """
#         if tensor_proto.size(0) <= 1:
#             return torch.tensor(0.0, device=tensor_proto.device)

#         # ρ = variance between prototypes
#         rho = tensor_proto.var(dim=0).mean()

#         # σ
#         if semi_supervised:
#             sigma = 0.5 * (self.log_sigma_l + self.log_sigma_u)
#         else:
#             sigma = self.log_sigma_l

#         alpha = self.config.ALPHA  # CRP concentration hyperparameter

#         d = tensor_proto.size(1)

#         lam = 2 * sigma * torch.log(
#             alpha / ((1 + rho / sigma) ** (d / 2))
#         )

#         return lam

    
#     def delete_empty_clusters(
#         self,
#         tensor_proto: torch.Tensor,
#         prob: torch.Tensor,
#         radii: torch.Tensor,
#         targets: torch.Tensor,
#         eps: float = 1e-3
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Remove clusters with negligible probability mass
#         """
#         keep = prob.sum(dim=0) > eps

#         tensor_proto = tensor_proto[keep]
#         radii = radii[keep]
#         targets = targets[keep]

#         return tensor_proto, radii, targets

    
#     def loss(
#         self,
#         q_dist: torch.Tensor,   # [Nq, nClusters], distances from queries to prototypes
#         y_q: torch.Tensor,      # [Nq], query labels
#         targets: torch.Tensor   # [nClusters], cluster labels
#     ) -> torch.Tensor:
#         """
#         Masked multi-modal loss using your weighted_loss function.
#         Only closest cluster per class contributes, matching IMP Eq. 7.

#         Args:
#             q_dist: distances from queries to clusters [Nq, nClusters]
#             y_q: query labels [Nq]
#             targets: cluster labels of each cluster [nClusters]

#         Returns:
#             scalar loss
#         """
#         device = q_dist.device
#         n_q, n_clusters = q_dist.shape
#         classes = torch.unique(y_q)
        
#         # Initialize weights mask: only closest cluster per class will get 1
#         weights = torch.zeros_like(q_dist, device=device)  # [Nq, nClusters]

#         for c in classes:
#             mask = targets == c
#             if mask.sum() == 0:
#                 continue
#             class_dists = q_dist[:, mask]               # [Nq, num_class_clusters]
#             min_idx = class_dists.argmin(dim=1)        # closest cluster index per query
#             # Fill weights = 1 for the closest cluster
#             cluster_indices = torch.arange(n_q, device=device)
#             selected_cols = mask.nonzero(as_tuple=False).squeeze(1)[min_idx]
#             weights[cluster_indices, selected_cols] = 1.0

#         # Use negative distances as logits
#         logits = -q_dist

#         # Compute weighted loss per your function
#         return weighted_loss(logits, y_q, weights).mean()


#     def forward(self, sample, super_classes=False, unsupervised=True):
#         """
#         IMP forward pass.

#         Args:
#             episode: Episode object from _process_batch()
#             super_classes: whether to use superclass labels (not used here)

#         Returns:
#             loss: scalar tensor
#             output: dict with 'loss', 'acc', 'logits'
#         """

#         episode=self._process_batch(sample, super_classes=super_classes)
#         # --- Unpack episode and remove batch dimension ---
#         x_s = episode.x_train.squeeze(0)   # [Ns, ...]
#         y_s = episode.y_train.squeeze(0)   # [Ns]
#         x_q = episode.x_test.squeeze(0)    # [Nq, ...]
#         y_q = episode.y_test.squeeze(0)    # [Nq]

#         if episode.x_unlabel is not None:
#             x_u = episode.x_unlabel.squeeze(0)
#         else:
#             x_u = None

#         device = x_s.device

#         # --- Embeddings ---
#         z_s = self.encoder(x_s)            # [Ns, D]
#         z_q = self.encoder(x_q)            # [Nq, D]
#         z_all = z_s
#         if x_u is not None:
#             z_u = self.encoder(x_u)        # [Nu, D]
#             z_all = torch.cat([z_s, z_u], dim=0)

#         # --- Initialize prototypes per class ---
#         classes = torch.unique(y_s)
#         protos = torch.stack([z_s[y_s == c].mean(0) for c in classes])
#         targets = classes.clone()

#         # Initial cluster radii (learnable scalar broadcast)
#         radii = self.log_sigma_l.expand(len(classes))  # [nClusters]

#         # --- Estimate lambda for DP-means ---
#         lam = self.estimate_lambda(protos, semi_supervised=(x_u is not None))

#         # --- DP-means cluster assignment ---
#         for i, zi in enumerate(z_all):
#             dists = torch.cdist(zi.unsqueeze(0), protos).squeeze(0)  # [nClusters]

#             # Labeled points: only allow cluster in their class
#             if i < len(z_s):
#                 mask = targets == y_s[i]
#                 dists = torch.where(mask, dists, torch.full_like(dists, float("inf")))

#             # Create new cluster if distance > lambda
#             if dists.min() > lam:
#                 r = self.log_sigma_l if i < len(z_s) else self.log_sigma_u
#                 r = r.view(1)  # ensure 1D
#                 protos = torch.cat([protos, zi.unsqueeze(0)], dim=0)
#                 radii = torch.cat([radii, r], dim=0)

#                 if i < len(z_s):
#                     targets = torch.cat([targets, y_s[i:i+1]])
#                 else:
#                     targets = torch.cat([targets, torch.tensor([-1], device=device)])

#         # --- Soft assignment refinement for labeled support ---
#         d = torch.cdist(z_s, protos)                     # [Ns, nClusters]
#         prob = F.softmax(-d / radii, dim=1)              # [Ns, nClusters]
#         protos = (prob.unsqueeze(-1) * z_s.unsqueeze(1)).sum(0) / prob.sum(0).unsqueeze(-1)

#         # --- Query classification ---
#         q_dist = torch.cdist(z_q, protos)                # [Nq, nClusters]

#         logits = []
#         for c in classes:
#             mask = targets == c
#             logits.append(-q_dist[:, mask].min(dim=1).values)
#         logits = torch.stack(logits, dim=1)             # [Nq, nClasses]

#         loss = self.loss(q_dist, y_q, targets)

#         # --- Compute accuracy ---
#         preds = logits.argmax(dim=1)
#         acc_val = (preds == y_q).float().mean()

#         # --- Return in required format ---
#         return loss, {
#             'loss': loss,
#             'acc': acc_val,
#             'logits': logits[0]  # first query point logits
#         }




#     def forward_unsupervised(
#         self,
#         sample: Any,
#         super_classes: bool,
#         unlabel_lambda: float = 20.0,
#         num_cluster_steps: int = 5
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Fully unsupervised DP-means clustering
#         """
#         episode=self._process_batch(sample, super_classes=super_classes)
#         x_s = episode.x_train.squeeze(0)   # [Ns, ...]
#         y_s = episode.y_train.squeeze(0)   # [Ns, ...]
#         y_q = episode.y_test.squeeze(0)   # [Ns, ...]
#         x_q = episode.x_test.squeeze(0)    # [Nq, ...]
#         z_u = self.encoder(x_s)
#         z_q = self.encoder(x_q)

#         device = z_u.device

#         # --- Initialize first prototype and radii ---
#         protos = z_u[:1]                      # start with first embedding as cluster
#         radii = self.log_sigma_u.view(1)      # 1D radii tensor

#         # --- DP-means cluster assignment ---
#         for _ in range(num_cluster_steps):
#             for zi in z_u:
#                 dists = torch.cdist(zi.unsqueeze(0), protos).squeeze(0)  # [nClusters]
#                 if dists.min() > unlabel_lambda:
#                     protos = torch.cat([protos, zi.unsqueeze(0)], dim=0)
#                     radii = torch.cat([radii, self.log_sigma_u.view(1)], dim=0)

#             # Soft assignment refinement
#             d = torch.cdist(z_u, protos)                       # [Nu, nClusters]
#             prob = F.softmax(-d / radii, dim=1)               # soft assignments
#             protos = (prob.unsqueeze(-1) * z_u.unsqueeze(1)).sum(0) / prob.sum(0).unsqueeze(-1)
        
#         z_s = self.encoder(episode.x_train.squeeze(0))  # [Ns, D]
#         y_s = episode.y_train.squeeze(0)               # [Ns] superclasses

#         # Assign prototypes to nearest support embedding
#         proto_to_superclass = []
#         for proto in protos:
#             dists = torch.cdist(proto.unsqueeze(0), z_s).squeeze(0)
#             nearest_idx = dists.argmin()
#             proto_to_superclass.append(y_s[nearest_idx])
#         proto_to_superclass = torch.stack(proto_to_superclass)  # [nClusters]

#         # Compute query predictions
#         q_dist = torch.cdist(z_q, protos)
#         pred_cluster = q_dist.argmin(dim=1)
#         pred_superclass = proto_to_superclass[pred_cluster]
#         acc_val = (pred_superclass == y_q).float().mean()
#         pseudo_loss = - (prob * torch.log(prob + 1e-12)).sum(dim=1).mean()

#         # --- Return in the same dict structure as forward() ---
#         return pseudo_loss, {
#             'loss': pseudo_loss,           # no supervised loss
#             'acc': acc_val,         # superclass-level accuracy
#             'logits': None,         # still None (optional)
#             'prototypes': protos,
#             'assignments': prob
#         }

