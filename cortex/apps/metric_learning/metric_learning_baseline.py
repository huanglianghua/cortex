import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR

import cortex.modules as modules
import cortex.data as data
import cortex.losses as losses
from cortex.engine import Model
from cortex.data import miners


__all__ = ['MetricLearningBaseline']


class MetricLearningBaseline(Model):

    def __init__(self,
                 loss_name='ContrastiveLoss',
                 batch_size=128,
                 instances_per_batch=8,
                 num_features=512):
        # sanity check
        if loss_name in ['NPairLoss', 'NTXentLoss']:
            # these losses require a (C x 2) sampling
            assert instances_per_batch == 2
        assert hasattr(losses, loss_name)

        # initialize parameters
        super(MetricLearningBaseline, self).__init__()
        self.loss_name = loss_name
        self.batch_size = batch_size
        self.instances_per_batch = instances_per_batch
        self.num_features = num_features
        self.normalize_embeds = True if not loss_name in [
            'NPairLoss', 'LargeMarginSoftmaxLoss'] else False

        # build backbone (with frozen BN) and embedding layers
        self.backbone = modules.bninception(
            pretrained=True,
            norm_layer=modules.FrozenBatchNorm2d)
        self.head = nn.Linear(1024, self.num_features)
    
    def forward(self, imgs):
        x = self.backbone(imgs)
        x = self.head(x)
        if self.normalize_embeds:
            normalize_embeds = F.normalize(x, p=2, dim=1)
        return x
    
    def train_step(self, batch, state):
        # use 1 warmup epoch, freeze backbone parameters and train others
        if state.epoch == 1 and state.step == 1:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        elif state.epoch > 1 and state.step == 1:
            for param in self.backbone.parameters():
                param.requires_grad_(True)

        # run forward pass and compute loss
        imgs, labels = batch
        embeds = self.forward(imgs)
        loss = state.criterions(embeds, labels)

        return loss

    def val_step(self, batch, state):
        # run forward pass and return embeddings and labels
        # (for MetricLearningMetrics evaluation)
        imgs, labels = batch
        embeds = self.forward(imgs)
        return embeds, labels
    
    def test_step(self, batch, state):
        return self.val_step(batch, state)
    
    def build_dataloader(self, dataset, mode, state):
        assert mode in ['train', 'val', 'test']
        if mode == 'train':
            sampler = data.MPerClassSampler(
                dataset.labels,
                batch_size=self.batch_size,
                num_instances=self.instances_per_batch,
                seed=state.cfg.manual_seed)
            transforms = T.Compose([
                T.Resize(size=256),
                T.RandomResizedCrop(
                    size=227,
                    scale=(0.16, 1),
                    ratio=(0.75, 1.33)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(
                    mean=[128. / 255, 117. / 255, 104. / 255],
                    std=[1. / 255, 1. / 255, 1. / 255])])
            dataset.transforms = transforms
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=32 // state.cfg.gpus_per_machine,
                pin_memory=state.cuda)
        else:
            sampler = data.TestSampler(len(dataset))
            transforms = T.Compose([
                T.Resize(size=256),
                T.CenterCrop(size=227),
                T.ToTensor(),
                T.Normalize(
                    mean=[128. / 255, 117. / 255, 104. / 255],
                    std=[1. / 255, 1. / 255, 1. / 255])])
            dataset.transforms = transforms
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=32 // state.cfg.gpus_per_machine,
                pin_memory=state.cuda,
                drop_last=False)
        return dataloader

    def build_training(self, state):
        # build criterion
        criterions = self._build_criterion(self.loss_name, state)

        # build optimizer(s)
        optimizers = self._build_optimizer(self.loss_name, criterions, state)

        # build scheduler(s)
        schedulers = self._build_scheduler(self.loss_name, optimizers, state)

        return criterions, optimizers, schedulers
    
    def _build_criterion(self, loss_name, state):
        assert hasattr(losses, loss_name)
        if loss_name == 'ContrastiveLoss':
            criterion = losses.ContrastiveLoss(
                pos_margin=0,
                neg_margin=0.5,
                distance_power=1,
                avg_nonzero_only=True,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllPairsMiner())
        elif loss_name == 'TripletMarginLoss':
            criterion = losses.TripletMarginLoss(
                margin=0.01,
                distance_power=1,
                avg_nonzero_only=True,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllTripletsMiner())
        elif loss_name == 'NCALoss':
            criterion = losses.NCALoss(
                scale=32,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'ProxyNCALoss':
            criterion = losses.ProxyNCALoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                scale=8,
                regularization_weight=0,
                normalize_proxies=True,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'NPairLoss':
            criterion = losses.NPairLoss(
                regularization_weight=0.02,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllPairsMiner())
        elif loss_name == 'LiftedStructureLoss':
            criterion = losses.LiftedStructureLoss(
                neg_margin=1.0,
                distance_power=1,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllPairsMiner(remove_symmetry=False))
        elif loss_name == 'MultiSimilarityLoss':
            criterion = losses.MultiSimilarityLoss(
                shift=0.5,
                pos_scale=2,
                neg_scale=40,
                normalize_embeds=self.normalize_embeds,
                miner=miners.MultiSimilarityMiner(
                    margin=0.1,
                    normalize_embeds=self.normalize_embeds))
        elif loss_name == 'CircleLoss':
            criterion = losses.CircleLoss(
                margin=0.4,
                scale=80,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllTripletsMiner())
        elif loss_name == 'FastAPLoss':
            criterion = losses.FastAPLoss(
                num_bins=10,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllPairsMiner(remove_symmetry=False))
        elif loss_name == 'ArcFaceLoss':
            criterion = losses.ArcFaceLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                margin=30,
                scale=32,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'SNRContrastiveLoss':
            criterion = losses.SNRContrastiveLoss(
                pos_margin=0.01,
                neg_margin=0.2,
                avg_nonzero_only=True,
                regularization_weight=0.1,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllPairsMiner())
        elif loss_name == 'NormalizedSoftmaxLoss':
            criterion = losses.NormalizedSoftmaxLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                temperature=0.05,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'SoftTripleLoss':
            criterion = losses.SoftTripleLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                centers_per_class=10,
                scale_center=10,
                scale_class=20,
                margin=0.01,
                regularization_weight=0.2,
                normalize_centers=True,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'ProxyAnchorLoss':
            criterion = losses.ProxyAnchorLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                margin=0.1,
                scale=32,
                regularization_weight=0,
                normalize_proxies=True,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'TupletMarginLoss':
            criterion = losses.TupletMarginLoss(
                margin=0.1,
                scale=64,
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllPairsMiner(remove_symmetry=False))
        elif loss_name == 'CosFaceLoss':
            criterion = losses.CosFaceLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                margin=0.35,
                scale=64,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'AngularLoss':
            criterion = losses.AngularLoss(
                margin=45,  # angle margin in degree
                normalize_embeds=self.normalize_embeds,
                miner=miners.AllTripletsMiner())
        elif loss_name == 'LargeMarginSoftmaxLoss':
            criterion = losses.LargeMarginSoftmaxLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                margin=4,
                scale=1,
                normalize_weights=False,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'SphereFaceLoss':
            criterion = losses.SphereFaceLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                num_features=self.num_features,
                margin=4,
                scale=1,
                normalize_embeds=self.normalize_embeds,
                miner=None)
        elif loss_name == 'NTXentLoss':
            criterion = losses.NTXentLoss(
                temperature=0.1,
                normalize_embeds=True,
                miner=None)
        elif loss_name == 'MarginLoss':
            criterion = losses.MarginLoss(
                num_classes=len(state.train_loader.dataset.CLASSES),
                margin=0.2,
                init_boundary=1.2,
                regularization_weight=0.02,
                normalize_embeds=True,
                miner=miners.AllTripletsMiner())
        return criterion

    def _build_optimizer(self, loss_name, criterion, state):
        # initialize default optimizer
        optimizers = [optim.Adam(
            self.parameters(),
            lr=1e-5 * state.world_size,
            weight_decay=5e-5)]
        
        # modify and/or append optimizers
        if loss_name == 'ProxyNCALoss':
            optimizers.append(optim.Adam(
                criterion.parameters(),
                lr=1e-1 * state.world_size,
                weight_decay=0))
        elif loss_name in ['MultiSimilarityLoss', 'CircleLoss', 'FastAPLoss']:
            params = []
            for k, v in self.named_parameters():
                if not v.requires_grad:
                    continue
                lr_mult = 0.1 if k.startswith('backbone') else 1.0
                params += [{'params': [v], 'lr_mult': lr_mult}]
            optimizers[0] = optim.Adam(
                params,
                lr=3e-5 * state.world_size,
                weight_decay=0.0005)
        elif loss_name in ['ArcFaceLoss', 'NormalizedSoftmaxLoss',
                           'LargeMarginSoftmaxLoss', 'SphereFaceLoss',
                           'CosFaceLoss']:
            optimizers.append(optim.Adam(
                criterion.parameters(),
                lr=1e-3 * state.world_size,
                weight_decay=0))
        elif loss_name == 'ProxyAnchorLoss':
            optimizers[0] = optim.AdamW(
                self.parameters(),
                lr=1e-4 * state.world_size,
                weight_decay=1e-4)
            optimizers.append(optim.AdamW(
                criterion.parameters(),
                lr=1e-2 * state.world_size,
                weight_decay=1e-4))
        
        return optimizers

    def _build_scheduler(self, loss_name, optimizers, state):
        schedulers = []
        if loss_name in ['MultiSimilarityLoss', 'CircleLoss']:
            schedulers.append(MultiStepLR(
                optimizers[0],
                milestones=[40, 80],
                gamma=0.1))
        elif loss_name == 'ProxyAnchorLoss':
            schedulers += [
                StepLR(optimizers[0], step_size=10, gamma=0.1),
                StepLR(optimizers[1], step_size=10, gamma=0.1)]
        return schedulers
