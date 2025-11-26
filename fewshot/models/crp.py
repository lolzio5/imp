import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 

from fewshot.models.model_factory import RegisterModel
from fewshot.models.basic import *
from fewshot.models.utils import *
import fewshot.utils.data_utils
from fewshot.data.episode import Episode
import pdb

@RegisterModel("crp")
class CRPModel(Protonet):

    def __init__(self, config, data, dataset=None):
        super(CRPModel, self).__init__(config, data)

        self.base_distribution = (0.0*torch.randn(1, 1, config.dim)).to(DEVICE)

    def _add_cluster(self, nClusters, protos, counts, radii, cluster_type='unlabeled', ex = None, use_mean=False,nc=1.):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            counts: [B, nClusters] number of examples in each cluster
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = torch.zeros(bsize, 1).to(DEVICE)
        if ex is None:
            counts = torch.cat([counts, zero_count], dim=1)
        else:
            counts[0,-1] = 1
            counts = torch.cat([counts, zero_count], dim=1)

        d_radii = torch.ones(bsize, 1).to(DEVICE).detach()
        d_radii = d_radii*(torch.exp(self.log_sigma_u)+torch.exp(self.log_sigma_l))
        
        new_proto = self.base_distribution.clone().detach().to(DEVICE)
        if use_mean:
            new_proto = torch.mean(protos, dim=1).unsqueeze(0)
        
        if ex is not None:
            protos[:,-1,:] = torch.exp(self.log_sigma_l)*new_proto + torch.exp(self.log_sigma_u)*ex.unsqueeze(0).unsqueeze(0).to(DEVICE)/(torch.exp(self.log_sigma_l)+torch.exp(self.log_sigma_u)*nc)
        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, counts, radii

    def _compute_priors(self, counts):
        """
        Args:
            counts: [B, nClusters] number of elements in each cluster
        Returns:
            priors: [B, nClusters] crp prior
        """
        ALPHA = self.config.ALPHA
        nClusters = counts.size()[1]
        num_examples = torch.sum(counts, 1, keepdim=True)  
        crp_prior_old = torch.ones_like(counts)
        crp_prior_new = torch.tensor([ALPHA], dtype=torch.float32).to(DEVICE)
        crp_prior_new = torch.cat([crp_prior_new.unsqueeze(0)]*nClusters, 1)
        indices = counts.view(-1).nonzero()
        if torch.numel(indices) != 0:
            values = torch.masked_select(crp_prior_old, torch.gt(counts, 0.0))
            crp_priors = crp_prior_new.put_(indices, values) #if going for uniform, can switch values to torch.ones_like(values)
        else:
            crp_priors = crp_prior_new

        return crp_priors

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)

        eps = self.config.eps
        xs = batch.x_train
        xq = batch.x_test

        h_train = self._run_forward(xs)
        bsize = h_train.size()[0]
        h_test = self._run_forward(xq)

        y_train_np = batch.y_train.detach().cpu().numpy()

        nClusters = len(np.unique(y_train_np))
        kShot = int(batch.y_train.shape[1]/nClusters)
        prob_train = one_hot(batch.y_train, nClusters).to(DEVICE)

        protos = self._compute_protos(h_train, prob_train)
        radii = (torch.exp(self.log_sigma_l)*torch.exp(self.log_sigma_u)/(torch.exp(self.log_sigma_l)+torch.exp(self.log_sigma_u)*kShot)) * torch.ones(bsize, nClusters).to(DEVICE).detach()

        total_prob = prob_train
        mu0 = torch.mean(protos, dim=1).unsqueeze(0)

        counts = self._get_count(total_prob, soft=True)
        priors = self._compute_priors(counts)

        #cluster unlabeled data
        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = torch.cat([h_train, h_unlabel], dim=1)

            nClusters, protos, counts, radii = self._add_cluster(nClusters, protos, counts, radii)

            priors = self._compute_priors(counts)

            #add one more cluster to the total probability tracked
            total_prob = torch.cat([total_prob, torch.zeros(total_prob.size(0), total_prob.size(1), 1).to(DEVICE)],dim=2)
            prob_train = torch.cat([prob_train, torch.zeros(prob_train.size(0), prob_train.size(1), 1).to(DEVICE)],dim=2)

            for ii in range(self.config.num_cluster_steps):
                all_unlabel_probs = None
                for i, ex in enumerate(h_unlabel[0]):
                    prob_unlabel = assign_cluster_radii_crp(protos, ex.unsqueeze(0).unsqueeze(0), radii, priors)#, priors)

                    if all_unlabel_probs is None:
                        all_unlabel_probs = prob_unlabel.detach()
                    else:
                        all_unlabel_probs = torch.cat([all_unlabel_probs, prob_unlabel.detach()],dim=1) 

                    #adding a cluster!
                    if (prob_unlabel[0,0,-1].item() > eps):

                        nClusters, protos, counts, radii = self._add_cluster(nClusters, protos, counts, radii,ex=ex.detach(),nc=eps)
                        prob_train = torch.cat([prob_train, torch.zeros(prob_train.size(0), prob_train.size(1), 1).to(DEVICE)],dim=2)
                        all_unlabel_probs = torch.cat([all_unlabel_probs, torch.zeros(all_unlabel_probs.size()[0], all_unlabel_probs.size()[1],1).to(DEVICE)],dim=2)
                                                
                        priors = self._compute_priors(counts)

                    total_prob = torch.cat([prob_train.detach(), all_unlabel_probs],dim=1)
                    counts = self._get_count(total_prob, soft=True)
 
                    priors = self._compute_priors(counts)

                total_prob = torch.cat([prob_train.detach(), all_unlabel_probs.detach()],dim=1)
                counts = self._get_count(total_prob, soft=True)

                protos = self._compute_protos(h_all, total_prob)
                priors = self._compute_priors(counts)

        logits = compute_logits_radii_crp(protos, h_test, radii, priors)
        
        y_pred = np.argmax(logits.detach().cpu().numpy(), axis=2)

        loss = F.cross_entropy(logits.squeeze(), batch.y_test.squeeze())
        
        # compute accuracy as float
        acc_val = float((y_pred.squeeze() == batch.y_test.cpu().numpy().squeeze()).mean())

        return loss, {
            'loss': loss.item(),
            'acc': acc_val,
            'logits': logits.detach().cpu().numpy()
            }