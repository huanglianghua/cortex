import fire

import cortex.data as data
import cortex.apps as apps
import cortex.engine as engine


def train_metric_learning_baseline(dataset='CUB200',
                                   root_dir=None,
                                   criterion='ContrastiveLoss'):
    # setup datasets
    train_data = getattr(data, dataset)(
        root_dir=root_dir, subset='train')
    val_data = getattr(data, dataset)(
        root_dir=root_dir, subset='test')

    # setup model
    batch_size = 128
    instances_per_batch = 8
    if criterion in ['NPairLoss', 'NTXentLoss']:
        instances_per_batch = 2
    elif criterion in ['SoftTripleLoss']:
        batch_size = 96
    model = apps.MetricLearningBaseline(
        loss_name=criterion,
        batch_size=batch_size,
        instances_per_batch=instances_per_batch)
    
    # run training
    engine.run_train(
        model,
        train_data,
        val_data=val_data,
        val_metric=data.MetricLearningMetrics(normalize_embeds=True),
        best_indicator='top1',
        num_epochs=160 if dataset == 'StanfordOnlineProducts' else 80,
        distributed=False,
        gpus_per_machine=1,
        non_blocking=True,
        max_grad_norm=10,
        val_frequency=5,
        log_dir='runs/ml_baseline_{}_{}'.format(dataset, criterion))


if __name__ == '__main__':
    # usage: `python tools/train.py [func_name] --[param1] [value1]...`
    # e.g., `python tools/train.py train_metric_learning_baseline --criterion TripletMarginLoss`
    fire.Fire()
