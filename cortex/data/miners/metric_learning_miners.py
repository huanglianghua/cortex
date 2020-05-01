import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cortex.ops as ops


__all__ = ['Miner', 'AllPairsMiner', 'AllTripletsMiner',
           'HardTripletsMiner', 'MultiSimilarityMiner',
           'DistanceWeightedMiner']


class Miner(nn.Module):
    r"""Base class for online example mining in metric learning losses.

    Arguments:
        normalize_embeds (bool): Whether to L2-normalize the embeddings
            or not. (default: False)
    """
    def __init__(self, normalize_embeds=False):
        super(Miner, self).__init__()
        self.normalize_embeds = normalize_embeds
    
    def forward(self, l_embeds, l_labels, r_embeds=None, r_labels=None):
        # disable gradients in the mining process
        with torch.no_grad():
            # normalize embeddings if necessary
            if self.normalize_embeds:
                l_embeds = F.normalize(l_embeds, p=2, dim=1)
                if r_embeds is not None:
                    r_embeds = F.normalize(r_embeds, p=2, dim=1)
            
            # check r_embeds and r_labels
            if r_embeds is None or r_labels is None:
                assert r_embeds is None and r_labels is None
                r_embeds = l_embeds
                r_labels = l_labels
            
            # online example mining
            output = self.mine(l_embeds, l_labels, r_embeds, r_labels)
        return output

    def mine(self, l_embeds, l_labels, r_embeds, r_labels):
        raise NotImplementedError


class AllPairsMiner(Miner):
    r"""Mine all available anchor-positive and anchor-negative pairs.

    Arguments:
        remove_symmetry (bool): Whether to remove symmetrical positive
            and negative pairs (e.g., [xj, xi] is a symmetrical item of
            [xi, xj]) or not. (default: True)
    """
    def __init__(self, remove_symmetry=True):
        super(AllPairsMiner, self).__init__(normalize_embeds=False)
        self.remove_symmetry = remove_symmetry
    
    def mine(self, l_embeds, l_labels, r_embeds, r_labels):
        pos_mask = (l_labels.unsqueeze(1) == r_labels.unsqueeze(0))
        neg_mask = (l_labels.unsqueeze(1) != r_labels.unsqueeze(0))
        if l_labels is r_labels:
            # remove diagonal elements in pos_mask
            pos_mask &= ~torch.eye(
                pos_mask.size(0),
                dtype=pos_mask.dtype,
                device=pos_mask.device)
            if self.remove_symmetry:
                # remove repeated positive and negative pairs
                pos_mask.triu_()
                neg_mask.triu_()
        pos_pairs = pos_mask.nonzero()
        neg_pairs = neg_mask.nonzero()
        return pos_pairs[:, 0], pos_pairs[:, 1], \
            neg_pairs[:, 0], neg_pairs[:, 1]


class AllTripletsMiner(Miner):
    r"""Mine all available anchor-positive-negative triplets.
    """
    def __init__(self):
        super(AllTripletsMiner, self).__init__(normalize_embeds=False)

    def mine(self, l_embeds, l_labels, r_embeds, r_labels):
        pos_mask = (l_labels.unsqueeze(1) == r_labels.unsqueeze(0))
        neg_mask = (l_labels.unsqueeze(1) != r_labels.unsqueeze(0))
        if l_labels is r_labels:
            # remove diagonal elements in pos_mask
            pos_mask &= ~torch.eye(
                pos_mask.size(0),
                dtype=pos_mask.dtype,
                device=pos_mask.device)
        triplet_mask = (pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1))
        anchor_inds, pos_inds, neg_inds = triplet_mask.nonzero().t()
        return anchor_inds, pos_inds, neg_inds


class HardTripletsMiner(Miner):
    r"""Mine the hardest positive and negative samples for each anchor.
    """
    def __init__(self,
                 distance='Euclidean',
                 distance_power=1,
                 normalize_embeds=True):
        assert distance in ['cosine', 'Euclidean']
        self.distance = distance
        self.distance_power = distance_power
        super(HardTripletsMiner, self).__init__(
            normalize_embeds=normalize_embeds)
    
    def mine(self, l_embeds, l_labels, r_embeds, r_labels):
        # compute distance matrix of embeddings
        if self.distance == 'cosine':
            distmat = torch.matmul(l_embeds, r_embeds.t())
            if not self.normalize_embeds:
                distmat = distmat / F.normalize(
                    l_embeds, p=2, dim=1).unsqueeze(1)
                distmat = distmat / F.normalize(
                    r_embeds, p=2, dim=1).unsqueeze(0)
            distmat = 2 - distmat
        elif self.distance == 'Euclidean':
            distmat = torch.cdist(l_embeds, r_embeds, p=2)
            distmat = distmat.pow(self.distance_power)
        
        # find all positive and negative pairs
        pos_mask = (l_labels[:, None] == r_labels[None, :]).float()
        neg_mask = (l_labels[:, None] != r_labels[None, :]).float()
        if l_labels is r_labels:
            # remove diagonal elements in pos_mask
            pos_mask -= pos_mask.diag().diag()
        neg_mask[neg_mask == 0] = float('inf')

        # find the hardest positive for each sample
        pos_dist = distmat * pos_mask
        pos_dist[torch.isnan(pos_dist)] = 0
        hardest_pos_inds = pos_dist.argmax(dim=1)

        # find the hardest negative for each sample
        neg_dist = distmat * neg_mask
        neg_dist[torch.isnan(neg_dist)] = float('inf')
        hardest_neg_inds = neg_dist.argmin(dim=1)

        # ensure anchor/positive/negative indices are not overlapped
        anchor_inds = torch.arange(
            distmat.size(0), device=distmat.device)
        assert torch.all(anchor_inds != hardest_pos_inds)
        assert torch.all(anchor_inds != hardest_neg_inds)
        assert torch.all(hardest_pos_inds != hardest_neg_inds)

        # keep only rows where both positive and negative pairs exist
        keep = torch.any(pos_mask != 0, dim=1) & \
            torch.any(neg_mask != 0, dim=1)
        anchor_inds = anchor_inds[keep]
        pos_inds = pos_inds[keep]
        neg_inds = neg_inds[keep]

        return anchor_inds, pos_inds, neg_inds


class MultiSimilarityMiner(Miner):
    r"""Hard pairs miner used in MultiSimilarityLoss.
    """
    def __init__(self, margin=0.1, normalize_embeds=True):
        self.margin = margin
        super(MultiSimilarityMiner, self).__init__(
            normalize_embeds=normalize_embeds)
    
    def mine(self, l_embeds, l_labels, r_embeds, r_labels):
        # compute similarity matrix between embeddings
        simmat = torch.matmul(l_embeds, r_embeds.t())

        # positive and negative pair masks
        pos_mask = (l_labels.unsqueeze(1) == r_labels.unsqueeze(0))
        neg_mask = (l_labels.unsqueeze(1) != r_labels.unsqueeze(0))
        if l_labels is r_labels:
            # remove diagonal elements in pos_mask
            pos_mask &= ~torch.eye(
                pos_mask.size(0),
                dtype=pos_mask.dtype,
                device=pos_mask.device)
        
        # find the smallest positive similarities
        pos_sim = simmat * pos_mask.float()
        pos_sim[~pos_mask] = float('inf')
        min_ap_sim, _ = pos_sim.min(dim=1, keepdim=True)

        # find the largest negative similarities
        neg_sim = simmat * neg_mask.float()
        neg_sim[~neg_mask] = -float('inf')
        max_an_sim, _ = neg_sim.max(dim=1, keepdim=True)

        # find hard positive/negative pairs that violates constaints
        hard_pos_mask = pos_sim - self.margin < max_an_sim
        hard_neg_mask = neg_sim + self.margin > min_ap_sim

        # generate indices
        pos_pairs = hard_pos_mask.nonzero()
        neg_pairs = hard_neg_mask.nonzero()

        return pos_pairs[:, 0], pos_pairs[:, 1], \
            neg_pairs[:, 0], neg_pairs[:, 1]


class DistanceWeightedMiner(Miner):
    r"""Hard triplets miner used in MarginLoss. Sampling negatives with
        weights inverse propotional to the distance distribution.
    """
    def __init__(self,
                 min_dist=0.1,
                 max_dist=1.42,
                 normalize_embeds=True):
        self.min_dist = min_dist
        self.max_dist = max_dist
        super(DistanceWeightedMiner, self).__init__(
            normalize_embeds=normalize_embeds)

    def mine(self, l_embeds, l_labels, r_embeds, r_labels):
        # compute distance matrix between embeddings
        distmat = torch.cdist(l_embeds, r_embeds, p=2)
        if l_embeds is r_embeds:
            distmat = distmat + float('inf') * torch.eye(
                distmat.size(0), device=distmat.device)
        distmat = distmat.clamp(min=self.min_dist)

        # compute sampling weights for negative pairs
        # (inverse propotional to probabilities)
        neg_mask = (l_labels.unsqueeze(1) != \
            r_labels.unsqueeze(0)).float()
        dim = l_embeds.size(1)
        log_weights = -((dim - 2) * distmat.log() + \
            (dim - 3) / 2 * (1 - distmat.pow(2) / 4))
        weights = torch.exp(log_weights - \
            torch.max(log_weights, dim=1, keepdim=True)[0])
        weights = weights * neg_mask * (
            (distmat < self.max_dist).float())
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(1e-6)

        # compute label statistics
        u_labels, u_counts = torch.unique(l_labels, return_counts=True)

        # sample negative pairs
        neg_mask = torch.zeros_like(distmat, dtype=torch.bool)
        n = r_labels.size(0)
        u_labels, u_counts = torch.unique(r_labels, return_counts=True)
        for i, label in enumerate(l_labels):
            k = u_counts[u_labels == label].item()
            if l_labels is r_labels:
                k -= 1
            neg_inds_i = np.random.choice(
                n, k, p=weights[i].cpu().numpy())
            neg_mask[i, neg_inds_i] = 1
        
        # generate triplet indices
        pos_mask = (l_labels.unsqueeze(1) == r_labels.unsqueeze(0))
        if l_labels is r_labels:
            # remove diagonal elements in pos_mask
            pos_mask &= ~torch.eye(
                pos_mask.size(0),
                dtype=pos_mask.dtype,
                device=pos_mask.device)
        triplet_mask = (pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1))
        anchor_inds, pos_inds, neg_inds = triplet_mask.nonzero().t()

        return anchor_inds, pos_inds, neg_inds
