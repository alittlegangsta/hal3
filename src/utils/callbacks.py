# src/utils/callbacks.py
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np

class LRFinder(Callback):
    """
    学习率查找器回调函数。
    在训练开始时，从一个很小的值指数级增加学习率，并记录损失。
    """
    def __init__(self, start_lr, end_lr, steps):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.steps = steps
        self.lrs = []
        self.losses = []
        self.lr_multiplier = (end_lr / start_lr) ** (1 / steps)

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.lrs.append(lr)
        self.losses.append(logs['loss'])
        new_lr = lr * self.lr_multiplier
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        if new_lr > self.end_lr:
            self.model.stop_training = True

class OneCycleLR(Callback):
    """
    独轮学习率策略回调函数。
    """
    def __init__(self, max_lr, steps_per_epoch, epochs, pct_start=0.3):
        super().__init__()
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.pct_start = pct_start
        self.lrs = []

    def on_train_begin(self, logs=None):
        self.total_steps = self.steps_per_epoch * self.epochs
        self.step_up = int(self.total_steps * self.pct_start)
        self.step_down = self.total_steps - self.step_up

    def on_train_batch_begin(self, batch, logs=None):
        current_step = self.model.optimizer.iterations.numpy()
        if current_step < self.step_up:
            # Phase 1: Increase LR
            p = current_step / self.step_up
            lr = self.max_lr * (0.05 + 0.95 * p)
        else:
            # Phase 2: Decrease LR
            p = (current_step - self.step_up) / self.step_down
            lr = self.max_lr * (1 - p) * (1 - 0.05) + self.max_lr * 0.05
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)