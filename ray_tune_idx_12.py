import os
from functools import partial

from ray import tune as ray_tune
from ray import train as ray_train
from ray.train import Checkpoint, get_checkpoint

from hyperopt import hp

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter

from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from util.util import *
from fixmatch import set_seed
from fixmatch import trainer as fixmatch_trainer

globally_best_iou = 0

def main(prev_best_cfgs, param_space, gpus_per_trial):
    set_seed(42)

    args = get_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.search_alg == 'rand':
        print("USING RAND SEARCH")
        
        search_alg = None
        
        scheduler = ASHAScheduler(
            max_t=args.num_epochs,
            grace_period=3,
            reduction_factor=2
        )

    elif args.search_alg == 'bohb':
        print("USING BOHB SEARCH")

        bohb = TuneBOHB(
            # param_space,
            # metric="main/grand_loss",
            # mode="min",
            points_to_evaluate=prev_best_cfgs
        )

        search_alg = ConcurrencyLimiter(bohb, max_concurrent=2)

        scheduler = HyperBandForBOHB(
            max_t=args.num_epochs,
            reduction_factor=3,
        )

    elif args.search_alg == 'hyperopt':
        print("USING HYPEROPT SEARCH")

        hyperopt = HyperOptSearch(
            param_space,
            metric="main/grand_loss",
            mode="min",
            points_to_evaluate=prev_best_cfgs,
        )

        search_alg = ConcurrencyLimiter(hyperopt, max_concurrent=2)

        scheduler = ASHAScheduler(
            max_t=args.num_epochs,
            grace_period=3,
            reduction_factor=2
        )
    
    tuner = ray_tune.Tuner(
        ray_tune.with_resources(
            ray_tune.with_parameters(partial(fixmatch_trainer, ray_train, args)),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=ray_tune.TuneConfig(
            metric="main/grand_loss",
            mode="min",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
        run_config=ray_train.RunConfig(
            storage_path=args.save_path,
            checkpoint_config=ray_train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="main/grand_loss",
            )
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    best_result = results.get_best_result("main/grand_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial grand loss: {}".format(best_result.metrics["main/grand_loss"]))
    print("Best trial final training loss: {}".format(best_result.metrics["epoch_train/loss"]))
    print("Best trial final validation loss: {}".format(best_result.metrics["eval/loss"]))
    print("Best trial final validation wIoU: {}".format(best_result.metrics["eval/wIoU"]))


if __name__ == "__main__":
    prev_best_cfgs = [
        {
            'class_weights_idx_2': 0.05,
            
            'lr': 0.0006641655771494509, 
            'weight_decay': 6.090603815955617e-07, 

            'conf_thresh': 0.5403738832570637, 
            'p_jitter': 0.7497370506266263, 
            'p_gray': 0.42058359750430085, 
            'p_blur': 0.6580713163389718, 
            'p_cutmix': 0.5427814390069718,

            'output_thresh': 0.6
        }
    ]

    param_space = {
        'grand_loss_weights': [1.0, 2.0, 4.0], 
        'crop_size': 800, 
        'batch_size': 2, 
        'unlabeled_ratio': 90.0, 
        
        'backbone': 'efficientnet-b0',
        # ray_tune.choice(['efficientnet-b0', 'timm-efficientnet-b0', 'efficientnet-b2', 'timm-efficientnet-b2', 'efficientnet-b3', 'timm-efficientnet-b3']),
        
        'class_weights_idx_2': ray_tune.uniform(0.05, 0.20),
        'class_weights': [0.008, 1.0, 0.048], 
        
        'loss_fn': 'cross_entropy',  # ray_tune.choice(['cross_entropy', 'jaccard', 'dice']),

        'lr': ray_tune.loguniform(1e-5, 1e-3),
        'lr_multi': 10.0,
        'weight_decay': ray_tune.loguniform(1e-9, 1e-5),
        
        'scheduler': 'poly', 
        
        'use_data_normalization': True,

        'conf_thresh': ray_tune.loguniform(0.5, 0.99),
        'p_jitter': ray_tune.uniform(0.0, 0.8),
        'p_gray': ray_tune.uniform(0.0, 0.8),
        'p_blur': ray_tune.uniform(0.0, 0.8),
        'p_cutmix': ray_tune.uniform(0.0, 0.8),

        'output_thresh': ray_tune.uniform(0.5, 0.95),
    }

    main(None, param_space, gpus_per_trial=0.5)