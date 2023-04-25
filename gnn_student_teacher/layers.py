import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer


class ExplanationSparsityRegularization(GraphBaseLayer):

    def __init__(self, factor: float = 1.0, **kwargs):
        super(ExplanationSparsityRegularization, self).__init__(**kwargs)
        self.factor = factor

    def call(self, inputs):
        # importances: ([batch], [N], K)
        importances = inputs

        loss = tf.reduce_mean(tf.math.abs(importances))
        self.add_loss(loss * self.factor)
