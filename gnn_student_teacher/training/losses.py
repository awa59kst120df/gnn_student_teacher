import tensorflow as tf
import tensorflow.keras as ks


def shifted_sigmoid(x, multiplier: float = 1, shift: float = 10) -> float:
    return ks.backend.sigmoid(multiplier * x - shift)


def bce(y_true, y_pred):
    """
    Binary Cross Entropy Loss. Implementation is based on the implementation that can be found within the
    keras library.

    You might be asking yourself why a custom implementation is needed if such a basic loss implementation
    also exists in the keras library. The reason for this is a bug in keras, where their default
    implementation for bce loss uses an operation which is not supported for RaggedTensors on GPUs. This
    bug causes errors when attempting to train a ragged tensor model with BCE loss on GPU architecture.
    This bug does not occur with this custom implementation
    """
    y_true = tf.cast(y_true, tf.float32)
    loss = y_true * tf.math.log(y_pred + 1e-7)
    loss += (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
    loss = -loss
    return tf.reduce_mean(loss)


class NoLoss(ks.losses.Loss):

    def __init__(self):
        ks.losses.Loss.__init__(self)
        self.name = 'no_loss'

    def call(self, y_true, y_pred):
        return 0.0


class ExplanationLoss(ks.losses.Loss):

    def __init__(self,
                 loss_function: ks.losses.Loss = bce,
                 mask_empty_explanations: bool = False,
                 reduce: bool = False,
                 factor: float = 1):
        ks.losses.Loss.__init__(self)

        self.loss_function = loss_function
        self.mask_empty_explanations = mask_empty_explanations
        self.factor = factor
        self.reduce = reduce
        self.name = 'explanation_loss'

    def call(self, y_true, y_pred):
        loss = self.loss_function(y_true, y_pred)

        if self.mask_empty_explanations:
            mask = tf.cast(tf.reduce_max(y_true, axis=-1) > 0, dtype=tf.float32)
            loss *= mask

        if self.reduce:
            loss = tf.reduce_mean(loss, axis=-1)

        return self.factor * loss
