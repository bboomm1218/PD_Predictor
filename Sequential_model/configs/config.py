from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from easydict import EasyDict as edict


configs = {

    'batch_size' : tune.choice([8, 16, 32, 64]),
    'hidden_size' : tune.choice([16, 32, 64]),
    'num_layers' : tune.choice([2, 3, 4]),
    'lr' : tune.loguniform(1e-4, 1e-1)
}

configs = edict(configs)

