import os
import json
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.models.gnnx import gnnx_importances
from graph_attention_student.models.gradient import GcnGradientModel, grad_importances
from graph_attention_student.training import mse, NoLoss
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == DATASET PARAMETERS ==
TEST_INDICES_PATH: str = os.path.join(ASSETS_PATH, 'mutagenicity_exp_test_indices.json')
# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/mutagenicity_exp')
TRAIN_RATIO: float = 0.85
NUM_EXAMPLES: int = 100
EXPLANATION_CHANNELS: int = 2
NUM_TARGETS: int = 1
SEED: int = 1

# == MODEL PARAMETERS ==


UNITS: t.List[int] = [16, 16, 16]
FINAL_UNITS: t.List[int] = [16, 8, 2]
FINAL_ACTIVATION: str = 'softmax'

# == TRAINING PARAMETERS ==
LOSS_CB = lambda: ks.losses.CategoricalCrossentropy()
EPOCHS: int = 15
BATCH_SIZE: int = 25
GNNX_EPOCHS: int = 300

# == GENERATION PARAMETERS ==
REPETITIONS: int = 3
CONSENSUS_RATIO: float = 0.5
POSTFIX: str = 'gnnx'

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_create_model_explanations.py')
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
BASE_PATH = PATH
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('overwrite_test_indices')
    def overwrite_test_indices(e, dataset, test_indices):
        with open(TEST_INDICES_PATH, mode='r') as file:
            content = file.read()
            indices = json.loads(content)

        e.info(f'loaded canonical test indices from file: {TEST_INDICES_PATH}')
        e.info(f'loaded {len(indices)} test indices')

        return indices


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
    def explain(e, model, x, y):
        node_input, edge_input, edge_indices = x
        num_elements = len(node_input.numpy())
        e.info(f'creating GNNExplainer explanations for {num_elements} elements...')

        # out_pred = model(
        #     x,
        #     training=False,
        #     batch_size=num_elements,
        #     create_gradients=False,
        #     return_gradients=False
        # )
        # e.info(f'made prediction. Starting to create explanations...')

        y_0 = ragged_tensor_from_nested_numpy([[1.0, 0.0] for _ in range(num_elements)])
        y_0 = tf.cast(y_0, tf.float32)
        ni0, ei0 = gnnx_importances(
            model, x, y_0,
            epochs=GNNX_EPOCHS,
            learning_rate=0.01,
            model_kwargs={
                'training': False,
                'batch_size': num_elements,
                'create_gradients': False,
                'return_gradients': False,
            },
            logger=e.logger,
            log_step=20,
        )

        y_1 = ragged_tensor_from_nested_numpy([[0.0, 1.0] for _ in range(num_elements)])
        y_1 = tf.cast(y_1, tf.float32)
        ni1, ei1 = gnnx_importances(
            model, x, y_1,
            epochs=GNNX_EPOCHS,
            learning_rate=0.01,
            model_kwargs={
                'training': False,
                'batch_size': num_elements,
                'create_gradients': False,
                'return_gradients': False,
            },
            logger=e.logger,
            log_step=20,
        )

        ni = tf.concat([ni0, ni1], axis=-1)
        ei = tf.concat([ei0, ei1], axis=-1)

        return ni, ei
