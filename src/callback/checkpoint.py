import os
import torch
from pytorch_lightning.utilities import rank_zero_only


class Checkpoint:
    """Legacy checkpoint helper retained for custom synchronized loggers."""
    def __init__(self, monitor: str, topk_models: int, output_path: str, exp_id = 'unknown_exp'):
        self.monitor = monitor
        self.best_models = list()
        self.topk_model = topk_models
        
        self.exp_id = exp_id
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok = True) 
    
    def save_model(self, model, score, epoch_idx=None):
        name_string = f'{self.exp_id}_epoch:{epoch_idx}_{self.monitor}:{score:.3f}.ckpt'

        torch.save(model.state_dict(), os.path.join(self.output_path, name_string))
    
    def remove_model(self, score, epoch_idx=None):
        name_string = f'{self.exp_id}_epoch:{epoch_idx}_{self.monitor}:{score:.3f}.ckpt'
        os.system(f'rm {os.path.join(self.output_path, name_string)}')
    
    def checkpointing(self, metric: dict, model, epoch_idx) -> None:
        if len(self.best_models) < self.topk_model:
            self.best_models.append((metric[self.monitor], epoch_idx))
            self.save_model(model, metric[self.monitor], epoch_idx)
        else:
            current_score = metric[self.monitor]
            for i, (score, idx) in enumerate(self.best_models):
                if current_score > score:
                    self.remove_model(score, idx)
                    self.best_models[i] = (current_score, epoch_idx)
                    self.save_model(model, current_score, epoch_idx)
                    break
    
    @rank_zero_only
    def __call__(self, metric: dict, model, epoch_idx, *args, **kwargs) -> None:
        self.checkpointing(metric=metric, model=model, epoch_idx=epoch_idx)
