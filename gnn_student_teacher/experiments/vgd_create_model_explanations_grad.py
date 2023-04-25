import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.models.gradient import GcnGradientModel, grad_importances
from graph_attention_student.training import mse, NoLoss
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingNodes

PATH = pathlib.Path(__file__).parent.absolute()

# == MODEL PARAMETERS ==
# All the parameters which determine the behavior of the MEGAN model
UNITS: t.List[int] = [16, 16, 16]
FINAL_UNITS: t.List[int] = [16, 8, 1]
FINAL_ACTIVATION: str = 'linear'

# == GENERATION PARAMETERS ==
REPETITIONS: int = 5
CONSENSUS_RATIO: float = 0.25
POSTFIX: str = 'grad'

# == TRAINING PARAMETERS ==
EPOCHS: int = 100
BATCH_SIZE: int = 32
LOSS_CB = lambda: mse

# == EVALUATION PARAMETERS ==
VAL_METRIC_KEY: str = 'val_mean_squared_error'

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_create_model_explanations.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('create_model')
    def create_model(e):
        model = GcnGradientModel(
            batch_size=e.p['BATCH_SIZE'],
            units=e.p['UNITS'],
            final_units=e.p['FINAL_UNITS'],
            final_activation=e.p['FINAL_ACTIVATION'],
            layer_cb=lambda k: AttentionHeadGATV2(
                k,
                use_edge_features=True,
                activation='kgcnn>leaky_relu',
            )
        )
        return model

    @se.hook('explain', replace=True)
    def explain(e, model: GcnGradientModel, x, y):
        node_input, edge_input, edge_indices = x
        num_elements = len(node_input.numpy())
        e.info(f'creating gradient explanations for {num_elements} elements...')

        out, ni_info, ei_info = model(
            x,
            training=False,
            batch_size=num_elements,
            create_gradients=True,
            return_gradients=True
        )
        ni = grad_importances(ni_info, use_relu=True)
        ei = grad_importances(ei_info, use_relu=True)

        return ni, ei
