import os
import sys
import torch
from omegaconf import OmegaConf

import utils.pytorch_utils as ptu
from learner import Learner


if __name__ == "__main__":

    env_cfg = OmegaConf.load("./config/env_config/dcc.yaml")

    ptu.set_gpu_mode(torch.cuda.is_available(), gpu_id=int(sys.argv[1]))
    # endregion

    algo_cfg = OmegaConf.load("./config/algo_config/mappo.yaml")
    expt_cfg = OmegaConf.load("./config/expt.yaml")
    cfg = OmegaConf.merge(env_cfg, algo_cfg, expt_cfg)

    print("cuda is available: ", torch.cuda.is_available())
    torch.set_num_threads(cfg.n_training_threads)
    os.makedirs(cfg.main_save_path, exist_ok=True)

    cfg.log_wandb = False
    cfg.save_model = False
    # cfg.use_recurrent_policy = True

    cfg.seed = 0

    learner = Learner(cfg)
    learner.train()

