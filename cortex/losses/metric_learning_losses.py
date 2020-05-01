import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import binom

import cortex.data.miners as miners
import cortex.ops as ops


__all__ = ['ContrastiveLoss', 'TripletMarginLoss', 'NCALoss',
           'ProxyNCALoss', 'NPairLoss', 'LiftedStructureLoss',
           'MultiSimilarityLoss', 'CircleLoss', 'FastAPLoss',
           'ArcFaceLoss', 'SNRContrastiveLoss', 'NormalizedSoftmaxLoss',
           'SoftTripleLoss', 'ProxyAnchorLoss', 'TupletMarginLoss',
           'CosFaceLoss', 'AngularLoss', 'LargeMarginSoftmaxLoss',
           'SphereFaceLoss', 'NTXentLoss', 'MarginLoss']


class ContrastiveLoss(nn.Module):
    r"""Contrastive loss published at CVPR'06.

    Publication:
        "Dimensionality Reduction by Learning an Invariant Mapping,"
            R. Hadsell, S. Chopra, and Y. Lecun. IEEE CVPR 2006.
    """
    def __init__(self,
                 pos_margin=0,
                 neg_margin=0.5,
                 distance='Euclidean',
                 distance_power=1,
                 avg_nonzero_only=True,
                 normalize_embeds=True,
                 miner=miners.AllPairsMiner()):
        assert distance in ['cosine', 'Euclidean']
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = distance
        self.distance_power = distance_power
        self.avg_nonzero_only = avg_nonzero_only
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds or self.distance == 'cosine':
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        a1_inds, pos_inds, a2_inds, neg_inds = self.miner(embeds, labels)
        if len(a1_inds) == 0 and len(a2_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[a1_inds] == labels[pos_inds])
        assert torch.all(labels[a2_inds] != labels[neg_inds])

        if self.distance == 'cosine':
            # compute similarity matrix of embeddings
            simmat = torch.matmul(embeds, embeds.t())
            if self.distance_power != 1:
                simmat = simmat.pow(self.distance_power)
            # compute anchor-positive and anchor-negative similarities
            ap_sim = simmat[a1_inds, pos_inds]
            an_sim = simmat[a2_inds, neg_inds]
            # compute anchor-positive and anchor-negative losses
            pos_loss = F.relu(self.pos_margin - ap_sim)
            neg_loss = F.relu(an_sim - self.neg_margin)
        elif self.distance == 'Euclidean':
            # compute distance matrix of embeddings
            distmat = torch.cdist(embeds, embeds, p=2)
            if self.distance_power != 1:
                distmat = distmat.pow(self.distance_power)
            # compute anchor-positive and anchor-negative distances
            ap_dist = distmat[a1_inds, pos_inds]
            an_dist = distmat[a2_inds, neg_inds]
            # compute anchor-positive and anchor-negative losses
            pos_loss = F.relu(ap_dist - self.pos_margin)
            neg_loss = F.relu(self.neg_margin - an_dist)
        else:
            raise ValueError(
                'Expected distance to be one of "cosine" or ' \
                'Euclidean, but got {}'.format(self.distance))
        
        # average over losses
        if self.avg_nonzero_only:
            pos_loss = pos_loss.sum() / (
                pos_loss.gt(0).sum().float() + 1e-12)
            neg_loss = neg_loss.sum() / (
                neg_loss.gt(0).sum().float() + 1e-12)
        else:
            pos_loss = pos_loss.mean()
            neg_loss = neg_loss.mean()
        
        return pos_loss + neg_loss


class TripletMarginLoss(nn.Module):
    r"""Triplet margin loss published at NIPS'06.

    Publication:
        "Distance Metric Learning for Large Margin Nearest Neighbor Classification,"
            K. Q. Weinberger, J. Blitzer, L. K. Saul. NIPS 2006.
    """
    def __init__(self,
                 margin=0.05,
                 distance_power=1,
                 avg_nonzero_only=True,
                 normalize_embeds=True,
                 miner=miners.AllTripletsMiner()):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.distance_power = distance_power
        self.avg_nonzero_only = avg_nonzero_only
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        anchor_inds, pos_inds, neg_inds = self.miner(embeds, labels)
        if len(anchor_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[anchor_inds] == labels[pos_inds])
        assert torch.all(labels[anchor_inds] != labels[neg_inds])

        # compute anchor-positive and anchor-negative distances
        distmat = torch.cdist(embeds, embeds, p=2)
        if self.distance_power != 1:
            distmat = distmat.pow(self.distance_power)
        ap_dist = distmat[anchor_inds, pos_inds]
        an_dist = distmat[anchor_inds, neg_inds]
        loss = F.relu(ap_dist - an_dist + self.margin)

        # average over losses
        if self.avg_nonzero_only:
            loss = loss.sum() / (loss.gt(0).sum().float() + 1e-12)
        else:
            loss = loss.mean()
        
        return loss


class NCALoss(nn.Module):
    r"""Neighbourhood components analysis loss published at NIPS'04.

    Publication:
        "Neighbourhood Components Analysis,"
            S. Roweis, G. Hinton, and R. Salakhutdinov. NIPS 2004.
    """
    def __init__(self,
                 scale=32,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, 'NCALoss does not support miner'
        super(NCALoss, self).__init__()
        self.scale = scale
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # compute distance matrix, and set diagonal elements to Inf
        distmat = torch.cdist(embeds, embeds, p=2).pow(2)
        inds = torch.arange(len(embeds), device=distmat.device)
        distmat[inds, inds] = float('inf')

        # compute loss (neighborhood component analysis)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        prob = F.softmax(-self.scale * distmat, dim=1)
        prob = (prob * pos_mask).sum(dim=1)
        loss = -prob[prob != 0].log().mean()  # negative-log-likelihood

        return loss


class ProxyNCALoss(nn.Module):
    r"""ProxyNCA loss published at ICCV'17.

    Publication:
        "No Fuss Distance Metric Learning Using Proxies."
            Y. Movshovitz-Attias et. al. IEEE ICCV 2017.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 scale=8,
                 regularization_weight=0,
                 normalize_proxies=True,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, 'ProxyNCALoss does not support miner'
        super(ProxyNCALoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.scale = scale
        self.regularization_weight = regularization_weight
        self.normalize_proxies = normalize_proxies
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.proxies = nn.Parameter(torch.randn(
            num_classes, num_features))
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        proxies = self.proxies
        if self.normalize_proxies:
            proxies = F.normalize(proxies, p=2, dim=1)
        
        # compute distance matrix between embeds and proxies
        distmat = torch.cdist(embeds, proxies, p=2).pow(2)

        # compute loss (neighborhood component analysis)
        proxy_labels = torch.arange(
            self.num_classes, device=labels.device)
        pos_mask = (labels.unsqueeze(1) == \
            proxy_labels.unsqueeze(0)).float()
        prob = F.softmax(-self.scale * distmat, dim=1)
        prob = (prob * pos_mask).sum(dim=1)
        loss = -prob[prob != 0].log().mean()  # negative-log-likelihood

        # add regularization term
        if self.regularization_weight > 0:
            loss = loss + self.proxies.norm(p=2, dim=1).mean() * \
                self.regularization_weight

        return loss


class NPairLoss(nn.Module):
    r"""N-Pair loss published at NIPS'16.

    Publication:
        "Improved Deep Metric Learning with Multi-class N-pair Loss Objective."
            K. Sohn. NIPS 2016.
    """
    def __init__(self,
                 regularization_weight=0.02,
                 normalize_embeds=False,
                 miner=miners.AllPairsMiner()):
        super(NPairLoss, self).__init__()
        if normalize_embeds:
            ops.warning('Setting normalize_embeds=True is not ' \
                        'recommended for NPairLoss!')
        self.regularization_weight = regularization_weight
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # ensure a P-K (K=2) sampling
        _, counts = labels.unique(return_counts=True)
        assert torch.all(counts == 2)
        
        # online example mining
        anchor_inds, pos_inds, _, _ = self.miner(embeds, labels)
        if len(anchor_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[anchor_inds] == labels[pos_inds])
        assert len(labels[anchor_inds].unique()) == len(anchor_inds)

        # compute the similarity matrix and build target
        simmat = torch.matmul(embeds[anchor_inds], embeds[pos_inds].t())
        target = torch.arange(len(simmat), device=simmat.device)

        # compute the loss
        loss = F.cross_entropy(simmat, target)
        if self.regularization_weight > 0:
            # add l2 embedding regularization term
            loss = loss + embeds.norm(p=2, dim=1).mean() * \
                self.regularization_weight
        
        return loss


class LiftedStructureLoss(nn.Module):
    r"""Lifted structure loss published at CVPR'16.

    Publication:
        "Deep metric learning via lifted structured feature embedding."
            H. Oh Song, Y. Xiang, S. Jegelka, S. Savarese. IEEE CVPR 2016.
    """
    def __init__(self,
                 neg_margin=1.0,
                 distance_power=1,
                 normalize_embeds=True,
                 miner=miners.AllPairsMiner(remove_symmetry=False)):
        super(LiftedStructureLoss, self).__init__()
        self.neg_margin = neg_margin
        self.distance_power = distance_power
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        a1_inds, pos_inds, a2_inds, neg_inds = self.miner(embeds, labels)
        if len(a1_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[a1_inds] == labels[pos_inds])

        # compute distance matrix
        distmat = torch.cdist(embeds, embeds, p=2)
        if self.distance_power != 1:
            distmat = distmat.pow(self.distance_power)
        
        # compute positive loss
        pos_mask = torch.zeros_like(distmat, dtype=torch.bool)
        pos_mask[a1_inds, pos_inds] = 1
        pos_loss = ops.logsumexp(distmat, mask=pos_mask, dim=1)

        # compute negative loss
        neg_mask = torch.zeros_like(distmat, dtype=torch.bool)
        neg_mask[a2_inds, neg_inds] = 1
        neg_loss = ops.logsumexp(
            self.neg_margin - distmat, mask=neg_mask, dim=1)
        
        # average over losses
        loss = F.relu(pos_loss + neg_loss).mean()

        return loss


class MultiSimilarityLoss(nn.Module):
    r"""Mullti-similarity loss published at CVPR'19.

    Publication:
        "Multi-similarity Loss with General Pair Weighting for Deep Metric Learning."
            X. Wang, X. Han, W. Huang, D. Dong, M. R. Scott. IEEE CVPR 2019.
    """
    def __init__(self,
                 shift=0.5,
                 pos_scale=2,
                 neg_scale=40,
                 normalize_embeds=True,
                 miner=miners.MultiSimilarityMiner(margin=0.1)):
        super(MultiSimilarityLoss, self).__init__()
        self.shift = shift
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        a1_inds, pos_inds, a2_inds, neg_inds = self.miner(embeds, labels)
        if len(a1_inds) == 0 and len(a2_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[a1_inds] == labels[pos_inds])
        assert torch.all(labels[a2_inds] != labels[neg_inds])
        
        # compute similarity matrix between embeddings
        simmat = torch.matmul(embeds, embeds.t())

        # compute positive loss
        pos_mask = torch.zeros_like(simmat, dtype=torch.bool)
        pos_mask[a1_inds, pos_inds] = 1
        pos_loss = (1. / self.pos_scale) * ops.logsumexp(
            -self.pos_scale * (simmat - self.shift),
            mask=pos_mask, add_one=True, dim=1)
        
        # compute negative loss
        neg_mask = torch.zeros_like(simmat, dtype=torch.bool)
        neg_mask[a2_inds, neg_inds] = 1
        neg_loss = (1. / self.neg_scale) * ops.logsumexp(
            self.neg_scale * (simmat - self.shift),
            mask=neg_mask, add_one=True, dim=1)
        
        # average over losses
        loss = torch.mean(pos_loss + neg_loss)

        return loss


class CircleLoss(nn.Module):
    r"""Circle loss published at CVPR'20.

    Publication:
        "Circle Loss: A Unified Perspective of Pair Similarity Optimization."
            Y. Sun, C. Cheng, Y. Zhang, et. al. IEEE CVPR 2020.
    """
    def __init__(self,
                 margin=0.4,
                 scale=80,
                 normalize_embeds=True,
                 miner=miners.AllTripletsMiner()):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        anchor_inds, pos_inds, neg_inds = self.miner(embeds, labels)
        if len(anchor_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[anchor_inds] == labels[pos_inds])
        assert torch.all(labels[anchor_inds] != labels[neg_inds])

        # compute similarity matrix of embeddings
        simmat = torch.matmul(embeds, embeds.t())
        ap_sims = simmat[anchor_inds, pos_inds]
        an_sims = simmat[anchor_inds, neg_inds]

        # compute losses for all anchors
        losses = []
        for i in range(len(embeds)):
            mask = (anchor_inds == i)
            if not mask.any():
                continue
            ap_sim = ap_sims[mask]
            an_sim = an_sims[mask]
            with torch.no_grad():
                ap_alpha = F.relu(1 + self.margin - ap_sim)
                an_alpha = F.relu(an_sim + self.margin)
            ap_term = -self.scale * ap_alpha * (ap_sim - 1 + self.margin)
            an_term = self.scale * an_alpha * (an_sim - self.margin)
            losses.append(F.softplus(
                ap_term.logsumexp(dim=0) + an_term.logsumexp(dim=0)))
        
        # average over losses
        loss = sum(losses) / len(losses)

        return loss


class FastAPLoss(nn.Module):
    r"""Fast-AP loss published at CVPR'20.

    Publication:
        "Deep Metric Learning to Rank."
        F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. IEEE CVPR 2019.
    """
    def __init__(self,
                 num_bins=10,
                 normalize_embeds=True,
                 miner=miners.AllPairsMiner(remove_symmetry=False)):
        super(FastAPLoss, self).__init__()
        self.num_bins = num_bins
        self.num_edges = num_bins + 1
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        a1_inds, pos_inds, a2_inds, neg_inds = self.miner(embeds, labels)
        if len(a1_inds) == 0 and len(a2_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[a1_inds] == labels[pos_inds])
        assert torch.all(labels[a2_inds] != labels[neg_inds])

        # compute distance matrix between embeddings
        distmat = torch.cdist(embeds, embeds, p=2).pow(2)
        
        # compute histogram pulses
        max_val = 4. if self.normalize_embeds else distmat.max().item()
        interval = max_val / self.num_bins
        points = torch.linspace(0, max_val, steps=self.num_edges,
                                device=embeds.device).view(-1, 1, 1)
        pulse = F.relu(1 - (distmat - points).abs() / interval)

        # positive and negative histograms
        pos_mask = torch.zeros_like(distmat)
        pos_mask[a1_inds, pos_inds] = 1
        pos_hist = (pulse * pos_mask).sum(dim=2).t()
        neg_mask = torch.zeros_like(distmat)
        neg_mask[a2_inds, neg_inds] = 1
        neg_hist = (pulse * neg_mask).sum(dim=2).t()

        # cumulative distribution
        pos_cumdist = torch.cumsum(pos_hist, dim=1)
        pos_product = pos_hist * pos_cumdist
        cumdist = torch.cumsum(pos_hist + neg_hist, dim=1)
        safe_hist = (pos_product > 0) & (cumdist > 0)
        if safe_hist.sum() == 0:
            return (0. * embeds).sum()
        
        # compute FastAP
        fast_ap = torch.zeros_like(pos_hist)
        fast_ap[safe_hist] = pos_product[safe_hist] / cumdist[safe_hist]
        fast_ap = fast_ap.sum(dim=1)

        # average over positive pairs
        pos_num = pos_mask.sum(dim=1)
        safe_pos = (pos_num > 0)
        if safe_pos.sum() == 0:
            return (0. * embeds).sum()
        fast_ap = fast_ap[safe_pos] / pos_num[safe_pos]
        loss = (1 - fast_ap).mean()

        return loss


class ArcFaceLoss(nn.Module):
    r"""ArcFace loss published at CVPR'19.

    Publication:
        "Arcface: Additive Angular Margin Loss for Deep Face Recognition."
            J. Deng, J. Guo, N. Xue, S. Zafeiriou. IEEE CVPR 2019.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 margin=30,
                 scale=32,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, 'ArcFaceLoss does not support miner'
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.margin = margin
        self.scale = scale
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.weights = nn.Parameter(torch.randn(
            num_classes, num_features))
    
    def forward(self, embeds, labels):
        # normalize embeddings
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
            embed_norms = embeds.new_ones(embeds.size(0))
        else:
            embed_norms = embeds.norm(p=2, dim=1)
        
        # normalize weights
        weights = F.normalize(self.weights, p=2, dim=1)

        # compute cosine similarities between embeddings and weights
        simmat = torch.matmul(embeds, weights.t())
        if not self.normalize_embeds:
            simmat = simmat / embed_norms.unsqueeze(1)
        
        # compute cos(angle + margin) for positive pairs
        inds = torch.arange(len(embeds), device=embeds.device)
        pos_angles = torch.acos(
            simmat[inds, labels].clamp(-1 + 1e-12, 1 - 1e-12))
        pos_logits = torch.cos(pos_angles + math.radians(self.margin))

        # compute cross-entropy loss
        logits = simmat.clone()
        logits[inds, labels] = pos_logits
        loss = F.cross_entropy(logits * self.scale, labels)

        return loss


class SNRContrastiveLoss(nn.Module):
    r"""Signal-to-Noise Ratio Contrastive loss published at CVPR'19.

    Publication:
        "Signal-to-noise Ratio: A Robust Distance Metric for Deep Metric Learning."
            T. Yuan, W. Deng, J. Tang, Y. Tang, B. Chen. IEEE CVPR 2019.
    """
    def __init__(self,
                 pos_margin=0.01,
                 neg_margin=0.2,
                 avg_nonzero_only=True,
                 regularization_weight=0.1,
                 normalize_embeds=True,
                 miner=miners.AllPairsMiner()):
        super(SNRContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.avg_nonzero_only = avg_nonzero_only
        self.regularization_weight = regularization_weight
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        a1_inds, pos_inds, a2_inds, neg_inds = self.miner(embeds, labels)
        if len(a1_inds) == 0 and len(a2_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[a1_inds] == labels[pos_inds])
        assert torch.all(labels[a2_inds] != labels[neg_inds])

        # compute SNR distance matrix of embeddings
        distmat = torch.var(
            embeds.unsqueeze(1) - embeds.unsqueeze(0), dim=2) / \
            torch.var(embeds, dim=1, keepdim=True)

        # compute anchor-positive and anchor-negative losses
        ap_dist = distmat[a1_inds, pos_inds]
        an_dist = distmat[a2_inds, neg_inds]
        pos_loss = F.relu(ap_dist - self.pos_margin)
        neg_loss = F.relu(self.neg_margin - an_dist)

        # average over losses
        if self.avg_nonzero_only:
            pos_loss = pos_loss.sum() / (
                pos_loss.gt(0).sum().float() + 1e-12)
            neg_loss = neg_loss.sum() / (
                neg_loss.gt(0).sum().float() + 1e-12)
        else:
            pos_loss = pos_loss.mean()
            neg_loss = neg_loss.mean()
        loss = pos_loss + neg_loss
        
        # add regularization term
        if self.regularization_weight > 0:
            loss = loss + embeds.sum(dim=1).abs().mean() * \
                self.regularization_weight
        
        return loss


class NormalizedSoftmaxLoss(nn.Module):
    r"""Normalized softmax loss published at BMVC'19.

    Publication:
        "Classification is a Strong Baseline for Deep Metric Learning."
            A. Zhai, H. Y. Wu. BMVC 2019.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 temperature=0.05,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, \
            'NormalizedSoftmaxLoss does not support miner'
        super(NormalizedSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.temperature = temperature
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.weights = nn.Parameter(torch.randn(
            num_classes, num_features))
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        weights = F.normalize(self.weights, p=2, dim=1)
        logits = torch.matmul(embeds, weights.t()) / self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss


class SoftTripleLoss(nn.Module):
    r"""SoftTriple loss published at ICCV'19.

    Publication:
        "SoftTriple Loss: Deep Metric Learning without Triplet Sampling."
            Q. Qian, L. Shang, B. Sun, J. Hu, H. Li, R. Jin. IEEE ICCV 2019.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 centers_per_class=10,
                 scale_center=10,
                 scale_class=20,
                 margin=0.01,
                 regularization_weight=0.2,
                 normalize_centers=True,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, \
            'SoftTripleLoss does not support miner'
        super(SoftTripleLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.centers_per_class = centers_per_class
        self.scale_center = scale_center
        self.scale_class = scale_class
        self.margin = margin
        self.regularization_weight = regularization_weight
        self.normalize_centers = normalize_centers
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.centers = nn.Parameter(torch.randn(
            num_classes * centers_per_class, num_features))
        nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        centers = self.centers
        if self.normalize_centers:
            centers = F.normalize(self.centers, p=2, dim=1)
        
        # average over center similarities to obtain class similarities
        sim_centers = torch.matmul(embeds, centers.t())
        sim_centers = sim_centers.view(
            -1, self.num_classes, self.centers_per_class)
        prob = F.softmax(self.scale_center * sim_centers, dim=2)
        sim_classes = torch.sum(prob * sim_centers, dim=2)

        # compute cross-entropy loss
        margin = torch.zeros_like(sim_classes)
        inds = torch.arange(len(margin), device=margin.device)
        margin[inds, labels] = self.margin
        loss = F.cross_entropy(
            self.scale_class * (sim_classes - margin), labels)
        
        # add regularization term
        if self.regularization_weight > 0 and \
            self.centers_per_class > 1:
            # compute same class mask of self.centers
            inds = torch.arange(self.num_classes, device=embeds.device)
            inds = inds.repeat(
                self.centers_per_class, 1).t().contiguous().view(-1)
            same_class = (inds.unsqueeze(1) == inds.unsqueeze(0))
            same_class.triu_(diagonal=1)

            # compute regularization loss
            c_simmat = torch.matmul(centers, centers.t())
            reg_loss = 2. - 2. * c_simmat[same_class] + 1e-6
            reg_loss = reg_loss.sqrt().sum() / same_class.sum().float()
            loss = loss + reg_loss * self.regularization_weight
        
        return loss


class ProxyAnchorLoss(nn.Module):
    r"""Proxy anchor loss published at CVPR'20.

    Publication:
        "Proxy Anchor Loss for Deep Metric Learning."
            S. Kim, D. Kim, M. Cho, S. Kwak. IEEE CVPR 2020.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 margin=0.1,
                 scale=32,
                 regularization_weight=0,
                 normalize_proxies=True,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, 'ProxyAnchorLoss does not support miner'
        super(ProxyAnchorLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.margin = margin
        self.scale = scale
        self.regularization_weight = regularization_weight
        self.normalize_proxies = normalize_proxies
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        # initialize proxies
        self.proxies = nn.Parameter(torch.randn(
            num_classes, num_features))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
    
    def forward(self, embeds, labels):
        # normalize embeddings
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
            embed_norms = embeds.new_ones(embeds.size(0))
        else:
            embed_norms = embeds.norm(p=2, dim=1)
        
        # normalize proxies
        proxies = self.proxies
        if self.normalize_proxies:
            proxies = F.normalize(proxies, p=2, dim=1)
            proxy_norms = proxies.new_ones(proxies.size(0))
        else:
            proxy_norms = proxies.norm(p=2, dim=1)
        
        # compute cosine similarities between embeddings and proxies
        cosine_mat = torch.matmul(embeds, proxies.t())
        if not self.normalize_embeds:
            cosine_mat = cosine_mat / embed_norms.unsqueeze(1)
        if not self.normalize_proxies:
            cosine_mat = cosine_mat / proxy_norms.unsqueeze(0)
        
        # compute positive loss
        pos_mask = F.one_hot(labels, self.num_classes).byte()
        pos_loss = ops.logsumexp(
            -self.scale * (cosine_mat - self.margin),
            mask=pos_mask, add_one=True, dim=0)
        pos_loss = pos_loss.sum() / pos_mask.any(dim=0).sum()

        # compute negative loss
        neg_mask = 1 - pos_mask
        neg_loss = ops.logsumexp(
            self.scale * (cosine_mat + self.margin),
            mask=neg_mask, add_one=True, dim=0)
        neg_loss = neg_loss.sum() / neg_mask.any(dim=0).sum()

        # regularization term
        loss = pos_loss + neg_loss
        if self.regularization_weight > 0:
            loss = loss + self.proxies.norm(p=2, dim=1).mean() * \
                self.regularization_weight

        return pos_loss + neg_loss


class TupletMarginLoss(nn.Module):
    r"""Tuplet margin loss published at CVPR'20.

    Publication:
        "Deep Metric Learning with Tuplet Margin Loss."
            B. Yu, D. Tao. IEEE ICCV 2019.
    """
    def __init__(self,
                 margin=0.1,
                 scale=64,
                 normalize_embeds=True,
                 miner=miners.AllPairsMiner(remove_symmetry=False)):
        assert normalize_embeds == True, \
            'normalize_embeds in TupletMarginLoss should be True' \
            'since it uses cosine similarity'
        super(TupletMarginLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        a1_inds, pos_inds, a2_inds, neg_inds = self.miner(embeds, labels)
        if len(a1_inds) == 0 and len(a2_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[a1_inds] == labels[pos_inds])
        assert torch.all(labels[a2_inds] != labels[neg_inds])

        # compute similarities of positive and negative pairs
        simmat = torch.matmul(embeds, embeds.t())
        ap_sim = torch.cos(torch.acos(
            simmat[a1_inds, pos_inds] - self.margin))  # slack margin
        an_sim = simmat[a2_inds, neg_inds]

        # compute loss
        exponent = self.scale * (
            an_sim.unsqueeze(0) - ap_sim.unsqueeze(1))
        mask = (a1_inds.unsqueeze(1) == a2_inds.unsqueeze(0))
        loss = ops.logsumexp(
            exponent, mask=mask, add_one=True, dim=1)
        loss = loss.mean()

        return loss


class CosFaceLoss(nn.Module):
    r"""Cosine face loss published at CVPR'18.

    Publication:
        "Cosface: Large Margin Cosine Loss for Deep Face Recognition."
            H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, W. Liu. IEEE CVPR 2018.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 margin=0.35,
                 scale=64,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, 'CosFaceLoss does not support miner'
        super(CosFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.margin = margin
        self.scale = scale
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.weights = nn.Parameter(torch.randn(
            num_classes, num_features))
    
    def forward(self, embeds, labels):
        # normalize embeddings
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
            embed_norms = embeds.new_ones(embeds.size(0))
        else:
            embed_norms = embeds.norm(p=2, dim=1)
        
        # normalize weights
        weights = F.normalize(self.weights, p=2, dim=1)

        # compute cosine similarities between embeddings and weights
        simmat = torch.matmul(embeds, weights.t())
        if not self.normalize_embeds:
            simmat = simmat / embed_norms.unsqueeze(1)
        
        # compute logits (add a slack margin for positive similarities)
        logits = simmat.clone()
        inds = torch.arange(len(embeds), device=embeds.device)
        logits[inds, labels] = simmat[inds, labels] - self.margin
        logits = self.scale * logits

        # compute loss
        loss = F.cross_entropy(logits, labels)

        return loss


class AngularLoss(nn.Module):
    r"""Angular loss published at ICCV'17.

    Publication:
        "Deep Metric Learning with Angular Loss."
            J. Wang, F. Zhou, S. Wen, X. Liu, Y. Lin. IEEE ICCV 2017.
    """
    def __init__(self,
                 margin=45,  # angle margin in degree
                 normalize_embeds=True,
                 miner=miners.AllTripletsMiner()):
        super(AngularLoss, self).__init__()
        self.margin = margin
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        anchor_inds, pos_inds, neg_inds = self.miner(embeds, labels)
        if len(anchor_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[anchor_inds] == labels[pos_inds])
        assert torch.all(labels[anchor_inds] != labels[neg_inds])

        # compute anchor-positive, anchor-negative, and
        # positive-negative similarities
        simmat = torch.matmul(embeds, embeds.t())
        ap_sim = simmat[anchor_inds, pos_inds]
        an_sim = simmat[anchor_inds, neg_inds]
        pn_sim = simmat[pos_inds, neg_inds]

        # compute loss
        tan2_margin = math.tan(math.radians(self.margin)) ** 2
        exponent = 4 * tan2_margin * (an_sim + pn_sim) - \
            2 * (1 + tan2_margin) * ap_sim
        loss = ops.logsumexp(exponent, add_one=True, dim=0)

        return loss.mean()


class LargeMarginSoftmaxLoss(nn.Module):
    r"""Large-margin softmax loss published at ICML'16. Also known as L-Softmax loss.

    Publication:
        "Large-Margin Softmax Loss for Convolutional Neural Networks."
            W. Liu, Y. Wen, Z. Yu, M. Yang. ICML 2016.
    """
    def __init__(self,
                 num_classes,
                 num_features,
                 margin=4,
                 scale=1,
                 normalize_weights=False,
                 normalize_embeds=False,
                 miner=None):
        assert miner is None, \
            'LargeMarginSoftmaxLoss does not support miner'
        super(LargeMarginSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.margin = margin
        self.scale = scale
        self.normalize_weights = normalize_weights
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.weights = nn.Parameter(torch.randn(
            num_classes, num_features))
    
    def forward(self, embeds, labels):
        # normalize embeddings
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
            embed_norms = embeds.new_ones(embeds.size(0))
        else:
            embed_norms = embeds.norm(p=2, dim=1)
        
        # normalize weights
        weights = self.weights
        if self.normalize_weights:
            weights = F.normalize(weights, p=2, dim=1)
            weight_norms = weights.new_ones(weights.size(0))
        else:
            weight_norms = weights.norm(p=2, dim=1)
        
        # compute cosine similarities between embeddings and weights
        cosine_mat = torch.matmul(embeds, weights.t())
        if not self.normalize_embeds:
            cosine_mat = cosine_mat / embed_norms.unsqueeze(1)
        if not self.normalize_weights:
            cosine_mat = cosine_mat / weight_norms.unsqueeze(0)
        
        # compute cosine(margin * angle) for positive pairs
        inds = torch.arange(len(embeds), device=embeds.device)
        cosine_pos = cosine_mat[inds, labels]
        margin_pos = self._cosine_with_margin(
            cosine_pos, margin=self.margin)
        
        # compute \phi(angle) for positive pairs
        with torch.no_grad():
            angles = torch.acos(
                cosine_pos.clamp(-1 + 1e-8, 1 - 1e-8))
            k = (self.margin * angles / math.pi).floor()
        phi = ((-1) ** k) * margin_pos - (2 * k)

        # compute logits
        logits = cosine_mat * embed_norms.unsqueeze(1) * \
            weight_norms.unsqueeze(0)
        logits[inds, labels] = phi * weight_norms[labels] * embed_norms

        # compute cross-entropy loss
        loss = F.cross_entropy(self.scale * logits, labels)

        return loss
    
    def _cosine_with_margin(self, cosines, margin):
        r"""Trigonometric multiple-angle formula for computing
            cosine(margin * angle).
        """
        cosines = cosines.unsqueeze(1)

        # pre-compute multipliers
        n = margin // 2
        inds = torch.arange(
            n + 1, dtype=torch.float32, device=cosines.device)
        factors = cosines.new_tensor(
            [binom(margin, 2 * int(i)) for i in inds])
        powers = cosines.new_tensor([margin - 2 * i for i in inds])
        signs = cosines.new_tensor([(-1) ** i for i in inds])

        # compute cosine(margin * angles)
        powered_cosines = cosines ** powers
        powered_sines = (1 - cosines ** 2) ** inds
        terms = signs * factors * powered_cosines * powered_sines

        return torch.sum(terms, dim=1)


class SphereFaceLoss(LargeMarginSoftmaxLoss):
    r"""Sphere face loss published at CVPR'17. Also known as A-Softmax loss.

    Publication:
        "Sphereface: Deep hypersphere embedding for face recognition."
            W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, L. Song. IEEE CVPR 2017.
    """
    def __init__(self, **kwargs):
        kwargs.update({'normalize_weights': True})
        super(SphereFaceLoss, self).__init__(**kwargs)


class NTXentLoss(nn.Module):
    r"""Normalized temperature-scaled cross-entropy loss at ArXiv'20.

    Publication:
        "A Simple Framework for Contrastive Learning of Visual Representations."
            T. Chen, S. Kornblith, M. Norouzi, G. Hinton. ArXiv 2020.
    """
    def __init__(self,
                 temperature=0.1,
                 normalize_embeds=True,
                 miner=None):
        assert miner is None, 'NTXentLoss does not support miner'
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.normalize_embeds = normalize_embeds
        self.miner = miner
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # ensure a P-K (K=2) sampling
        _, counts = labels.unique(return_counts=True)
        assert torch.all(counts == 2)

        # compute similarity matrix of embeddings, scaled by temperature
        # (set diagonal elements to inf)
        simmat = torch.matmul(embeds, embeds.t()) / self.temperature
        inds = torch.arange(len(simmat), device=simmat.device)
        simmat[inds, inds] = -float('inf')
        
        # build target
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
        pos_mask[inds, inds] = 0
        nonzero = pos_mask.nonzero()
        assert torch.all(nonzero[:, 0] == torch.arange(
            len(nonzero), device=nonzero.device))
        target = nonzero[:, 1]

        # compute cross-entropy loss
        loss = F.cross_entropy(simmat, target)
        
        return loss


class MarginLoss(nn.Module):
    r"""Margin loss published at ICCV'17.

    Publication:
        "Sampling Matters in Deep Embedding Learning."
            C. Y. Wu, R. Manmatha, A. J. Smola, P. Krahenbuhl. IEEE ICCV 2017.
    """
    def __init__(self,
                 num_classes,
                 margin=0.2,
                 init_boundary=1.2,
                 regularization_weight=0.02,
                 normalize_embeds=True,
                 miner=miners.AllTripletsMiner()):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.init_boundary = init_boundary
        self.regularization_weight = regularization_weight
        self.normalize_embeds = normalize_embeds
        self.miner = miner
        self.boundaries = nn.Parameter(
            torch.ones(num_classes) * init_boundary)
    
    def forward(self, embeds, labels):
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        
        # online example mining
        anchor_inds, pos_inds, neg_inds = self.miner(embeds, labels)
        if len(anchor_inds) == 0:
            return (0. * embeds).sum()
        assert torch.all(labels[anchor_inds] == labels[pos_inds])
        assert torch.all(labels[anchor_inds] != labels[neg_inds])
        
        # compute anchor-positive and anchor-negative distances
        distmat = torch.cdist(embeds, embeds, p=2)
        ap_dist = distmat[anchor_inds, pos_inds]
        an_dist = distmat[anchor_inds, neg_inds]

        # compute losses of positive and negative pairs
        beta = self.boundaries[labels[anchor_inds]]
        pos_loss = F.relu(ap_dist - beta + self.margin)
        neg_loss = F.relu(beta - an_dist + self.margin)

        # average over non-zero losses
        loss = torch.cat([pos_loss, neg_loss], dim=0)
        loss = loss[loss > 0].mean()
        if self.regularization_weight > 0:
            loss = loss + beta.mean() * self.regularization_weight
        
        return loss
