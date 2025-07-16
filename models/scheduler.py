import math
import torch
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        if warmup > max_iters:
            raise ValueError(f"Given {warmup=} > {max_iters=}")
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.warmup:
            lr_factor = epoch / self.warmup
        elif epoch < self.max_num_iters:
            lr_factor = (1 + math.cos(math.pi*(epoch-self.warmup) \
                                      /(self.max_num_iters-self.warmup))) / 2
        else:
            lr_factor = 0
        return lr_factor

