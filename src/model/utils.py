from pytorch_lightning.utilities import rank_zero_only


class WanbSyncLogger:
    """Synchronized wandb logger. 
    The raw version's step counting is async, because of pl calls """
    def __init__(self, run, log_every_n_steps=None, step_suffix='_step', epoch_suffix='_epoch', log_backend='wandb'):
        self.run = run
        self.log_every_n_steps = log_every_n_steps

        if log_backend == 'wandb':
            import wandb

            self.log_backend = wandb
        elif log_backend == 'swanlab':
            import swanlab

            self.log_backend = swanlab
        else:
            raise ValueError(f'Invalid log backend specified: {log_backend}')

        self.cached_data = dict()
        self.current_step = 0
        self.old_current_epoch = 0
        self.current_epoch = 0
        
        self._step_suffix = step_suffix
        self._epoch_suffix = epoch_suffix

    @rank_zero_only
    def add_step(self):
        self.current_step += 1

    @rank_zero_only
    def add_epoch(self):
        self.current_epoch += 1

    @rank_zero_only
    def sync_epoch(self):
        self.old_current_epoch = self.current_epoch

    @rank_zero_only
    def log(self, data: dict, on_step=None, on_epoch=None, step=None):
        if not self.log_every_n_steps or (on_step is None and on_epoch is None):
            self.log_backend.log(data, step=self.current_step)
            return
        
        # if k in d, append v， else create [v]
        _dict_list_updater = lambda d, k, v: d[k].append(v) if k in d.keys() else d.update({k: [v]})
        # reduction by the `mean` of a list
        _list_reducer = lambda x: sum(x) / len(x)

        # on-step metrics
        if on_step:
            # 1. update
            for k, v in data.items():
                _dict_list_updater(self.cached_data, k + self._step_suffix, v)
            
            # 2. log & 3. reset
            if self.current_step % self.log_every_n_steps == 0:
                self.log_backend.log({k + self._step_suffix: _list_reducer(self.cached_data[k + self._step_suffix])
                            for k in data.keys()}, step=self.current_step)
                for k in data.keys():
                    self.cached_data[k + self._step_suffix] = []

        if on_epoch:
            # on-epoch metrics
            if self.current_epoch != self.old_current_epoch:
                epoch_keys = list(filter(lambda x: self._epoch_suffix in x, self.cached_data.keys()))
                self.log_backend.log({k: _list_reducer(self.cached_data[k]) for k in epoch_keys}, step=self.current_step)
                for k in epoch_keys:
                    self.cached_data[k] = []
            else:
                for k, v in data.items():
                    _dict_list_updater(self.cached_data, k + self._epoch_suffix, v)
