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
from supervised import set_seed
from supervised import trainer

globally_best_iou = 0

def main(prev_best_cfgs, param_space, gpus_per_trial, grace_period):
    set_seed(42)

    args = get_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.search_alg == 'rand':
        print("USING RAND SEARCH")
        
        search_alg = None
        
        scheduler = ASHAScheduler(
            max_t=args.num_epochs,
            grace_period=grace_period,
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
            # param_space,
            # metric="main/grand_loss",
            # mode="min",
            points_to_evaluate=prev_best_cfgs,
        )

        search_alg = ConcurrencyLimiter(hyperopt, max_concurrent=1)

        scheduler = ASHAScheduler(
            max_t=args.num_epochs,
            grace_period=grace_period,
            reduction_factor=2
        )
    
    tuner = ray_tune.Tuner(
        ray_tune.with_resources(
            ray_tune.with_parameters(partial(trainer, ray_train, args)),
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
    param_space = {
        'grand_loss_weights': [1.0, 2.0, 4.0],

        'crop_size': 800,
        'batch_size': 2,  # 2, 4, 8, 16

        'backbone': 'efficientnet-b0',
        'pretrained': True, # ray_tune.choice([False, True]),  # False, True

        'loss_fn': 'cross_entropy',  # 'cross_entropy', 'jaccard', 'combined'
        'lr': 2e-4, # ray_tune.choice([1e-4, 2e-4, 4e-4]),

        'lr_multi': 10.0,  # used only when pretrained is true
        'weight_decay': 1e-9,

        'scheduler': 'poly',

        'data_normalization': 'labeled',  # 'none', 'labeled', 'validation', 'unlabeled'

        'output_thresh' : 0.5, # ray_tune.choice([0.5, 0.7, 0.9]),

        'p_jitter_l': 0.0, # ray_tune.choice([0.0, 0.2, 0.4, 0.6]),
        'p_gray_l'  : ray_tune.choice([0.0, 0.25, 0.5]),
        'p_blur_l'  : ray_tune.choice([0.0, 0.25, 0.5]),
        'p_cutmix_l': 0.0, # ray_tune.choice([0.0, 0.25, 0.5]),
    }

    main(None, param_space, gpus_per_trial=1.0, grace_period=10)