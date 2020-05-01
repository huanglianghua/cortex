import fire

import cortex.data as data
import cortex.apps as apps
import cortex.engine as engine


def test_metric_learning_baseline(dataset='CUB200',
                                  root_dir=None,
                                  criterion='ContrastiveLoss'):
    test_data = getattr(data, dataset)(root_dir=root_dir, subset='test')
    model = apps.MetricLearningBaseline(
        loss_name=criterion,
        instances_per_batch=2 if criterion in [
            'NPairLoss', 'NTXentLoss'] else 8)
    engine.run_evaluation(
        model,
        test_data,
        test_metric=data.MetricLearningMetrics(normalize_embeds=True),
        distributed=False,
        gpus_per_machine=1,
        load_from='runs/ml_baseline_{}_{}/latest.pth'.format(
            dataset, criterion),
        log_dir='runs/ml_baseline_{}_{}_eval'.format(
            dataset, criterion))


if __name__ == '__main__':
    # usage: `python tools/test.py [func_name] --[param1] [value1]...`
    # e.g., `python tools/test.py test_metric_learning_baseline --criterion TripletMarginLoss`
    fire.Fire()
