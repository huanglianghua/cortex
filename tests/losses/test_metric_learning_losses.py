import unittest
import torch

import cortex.losses as losses


class TestMetricLearningLosses(unittest.TestCase):

    def setUp(self):
        self.num_classes = 100
        self.num_features = 512
        self.batch_size = 128
        
        # setup GPU if available
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')

    def test_pair_based_losses(self):
        loss_names = [
            'ContrastiveLoss', 'TripletMarginLoss', 'NCALoss',
            'NPairLoss', 'LiftedStructureLoss', 'MultiSimilarityLoss',
            'CircleLoss', 'FastAPLoss', 'SNRContrastiveLoss',
            'TupletMarginLoss', 'AngularLoss', 'NTXentLoss',
            'MarginLoss']
        for name in loss_names:
            # initialize random embeddings and labels
            embeds = torch.randn(
                self.batch_size, self.num_features).to(self.device)
            if name in ['NPairLoss', 'NTXentLoss']:
                labels = torch.arange(
                    self.batch_size // 2,
                    device=self.device).unsqueeze(1).repeat(1, 2)
            else:
                labels = torch.arange(
                    self.batch_size // 8,
                    device=self.device).unsqueeze(1).repeat(1, 8)
            labels = labels.view(-1)

            # initialize criterion
            if name == 'MarginLoss':
                criterion = getattr(losses, name)(
                    num_classes=self.num_classes).to(self.device)
            else:
                criterion = getattr(losses, name)().to(self.device)

            # compute loss and check sanity
            loss = criterion(embeds, labels)
            self.assertGreaterEqual(loss, 0)
    
    def test_proxy_based_losses(self):
        loss_names = [
            'ProxyNCALoss', 'ArcFaceLoss', 'NormalizedSoftmaxLoss',
            'SoftTripleLoss', 'ProxyAnchorLoss', 'CosFaceLoss',
            'LargeMarginSoftmaxLoss', 'SphereFaceLoss']
        for name in loss_names:
            criterion = getattr(losses, name)(
                num_classes=self.num_classes,
                num_features=self.num_features).to(self.device)
            embeds = torch.randn(
                self.batch_size, self.num_features).to(self.device)
            labels = torch.randint(
                100, (self.batch_size, ), device=self.device)
            loss = criterion(embeds, labels)
            self.assertGreaterEqual(loss, 0)


if __name__ == '__main__':
    unittest.main()
