import sys
import hydra
import torch
# import wandb
import swanlab
import logging
import pytorch_lightning as pl

from pathlib import Path
from datetime import datetime
# from pytorch_lightning.loggers import WandbLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path.cwd() / 'config')
CONFIG_NAME = 'main'
WANDB_API_KEY = open('wandb_api_key.txt', mode='r').read()
SWANLAB_API_KEY = open('swanlab_api_key.txt', mode='r').read()

def try_resume_training(exp):
    """specify an existing experiment name, them try resume training"""
    save_dir = Path(exp.save_dir).resolve().joinpath('../')  # back to top folder
    checkpoints = list(save_dir.glob(f'**/{exp.uuid.split('@')[0]}*/checkpoints/*.ckpt'))
    
    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None
    
    log.info('\nFound following check points:\n')
    for i, ckpt in enumerate(checkpoints):
        log.info(f'{i} - {ckpt}\n')
    log.info(f'Please select one checkpoint by its number.')
    
    ckpt_idx = timeout_input(timeout=10)
    ckpt_path = checkpoints[-1] if ckpt_idx is None else checkpoints[ckpt_idx]
    log.info(f'Use checkpoint: {ckpt_path}')
    
    return ckpt_path

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config):
    pl.seed_everything(config.exp.seed)
    ckpt_path = try_resume_training(config.exp)
    
    @rank_zero_only
    def _make_log():
        Path(config.exp.save_dir).mkdir(exist_ok=True, parents=True)

    dataset = setup_dataset(config)
    # terrible code:( have to compromise.
    config.runner.scheduler.scheduler_params.steps_per_epoch = len(dataset.train_dataloader())
    model = setup_model(config)
    
    if ckpt_path is None:
        run_id = config.exp.uuid 
    else:
        print(model.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False))
        run_id = f'{config.exp.uuid}@Re@{datetime.now().strftime("%m%d_%H%M%S")}@{config.exp.cmt}'
    
    logger = SwanLabLogger(
        project=config.exp.project,
        save_dir=config.exp.save_dir,
        experiment_name=run_id,
        tags=config.runner.tags,
    )
    
    callbacks = [GitDiffCallback(config), ModelCheckpoint(save_top_k=0)]
    # start training
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **config.runner.trainer)
    trainer.fit(model, datamodule=dataset)
    
    
if __name__ == '__main__':
    sys.path.append('./')
    from src.runner import setup_model, setup_dataset
    from src.callback.git_diff import GitDiffCallback
    from src.runner import timeout_input
    # wandb.login(key=WANDB_API_KEY)
    swanlab.login(api_key=SWANLAB_API_KEY)
    torch.set_float32_matmul_precision('medium')
    main()