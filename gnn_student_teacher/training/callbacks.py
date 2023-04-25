import time
import logging
import typing as t

import tensorflow as tf
import tensorflow.keras as ks


class EpochCounterCallback(ks.callbacks.Callback):

    def __init__(self):
        super(EpochCounterCallback, self).__init__()
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1


class LogProgressCallback(EpochCounterCallback):

    def __init__(self,
                 logger: logging.Logger,
                 key: t.Union[str, t.List[str]],
                 epoch_step: int):
        super(LogProgressCallback, self).__init__()

        self.logger = logger
        if isinstance(key, list):
            self.keys = key
        else:
            self.keys = [key]
        self.epoch_step = epoch_step

        self.start_time = time.time()
        self.elapsed_time = 0
        self.active = True

    def on_epoch_end(self, epoch, logs=None):
        if self.active:

            if self.epoch % self.epoch_step == 0 or self.epoch == 0:
                self.elapsed_time = time.time() - self.start_time

                string_list = [
                    f' * epoch {str(self.epoch):<5}',
                    f'elapsed_time={self.elapsed_time:.1f}s'
                ]
                for key in self.keys:
                    if key in logs:
                        value = logs[key]
                        string_list.append(f'{key}={value:.2f}')

                self.logger.info(' - '.join(string_list))

            super(LogProgressCallback, self).on_epoch_end(epoch, logs)
