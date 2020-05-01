import numpy as np
import faiss
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score

from .default_metrics import Metric


__all__ = ['MetricLearningMetrics']


class MetricLearningMetrics(Metric):

    def __init__(self,
                 normalize_embeds=False,
                 use_gpu=False):
        self.normalize_embeds = normalize_embeds
        self.use_gpu = use_gpu
        self.reset()
    
    def reset(self):
        self.computed = False
        self.embeds = []
        self.labels = []
    
    def update(self, output):
        self.computed = False
        embeds, labels = output
        self.embeds.append(embeds.cpu().numpy())
        self.labels.append(labels.cpu().numpy())

    def compute(self):
        if not self.computed:
            embeds = np.concatenate(self.embeds, axis=0)
            labels = np.concatenate(self.labels, axis=0)
            if self.normalize_embeds:
                embeds = normalize(embeds)

            # find the k-nearest neighbors using faiss
            u_labels, u_counts = np.unique(labels, return_counts=True)
            k = min(1023, int(u_counts.max()))
            index = faiss.IndexFlatL2(embeds.shape[1])
            if self.use_gpu and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(index)
            index.add(embeds)
            _, inds = index.search(embeds, k + 1)
            inds = inds[:, 1:]
            knn_labels = labels[inds]

            # compute the clustering quality
            num_clusters = len(set(labels))
            pred_labels = self._kmeans(embeds, num_clusters)
            nmi = normalized_mutual_info_score(labels, pred_labels)

            # compute top-k precisions
            nums = [1, 2, 4, 8, 16, 32]
            topk = [np.any(
                knn_labels[:, :k] == labels[:, None], axis=1).mean()
                for k in nums]

            # build output metrics
            self._metrics = {'nmi': nmi}
            for i, k in enumerate(nums):
                self._metrics.update({'top%d' % k: topk[i]})
        return self._metrics
    
    def _kmeans(self, x, k):
        n, c = x.shape

        # faise implementation of k-means
        cluster = faiss.Clustering(c, k)
        cluster.niter = 20
        cluster.max_points_per_centroid = 10000000
        index = faiss.IndexFlatL2(c)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
        
        # run training
        cluster.train(x, index)
        _, inds = index.search(x, 1)

        return [int(t[0]) for t in inds]
