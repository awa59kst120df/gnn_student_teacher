"""
This experiment conducts a student teacher analysis for the synthetic rb adversarial motifs dataset, where
the ratio of noisy explanations in the dataset is gradually increased during a sweep.

CHANGELOG

0.1.0 - 28.03.2023 - Initial version
"""
import os
import pathlib
import random
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.layers import CoefficientActivation

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/rb_adv_motifs')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = []
EXPLANATION_CHANNELS: int = 2
EXPLANATION_POSTFIX: str = '2'
# :param DATASET_SELECT_KEY:
#       Optionally this may specify a string metadata key for the VGD. The corresponding value for each
#       element of the dataset is supposed to be a boolean value. If that value is True then the item
#       will be considered part of the dataset, otherwise it won't. This is essentially a means to
#       pre-filter the dataset so that only a subset is used for the student teacher analysis.
DATASET_FILTER_KEY: t.Optional[str] = None  # model_explanations
# :param TRAIN_RATIO:
#       The ratio of the dataset to be used for the training of the students. The rest of it will
#       implicitly be used for the testing
TRAIN_RATIO: float = 0.8
# :param TRAIN_NUM:
#       Sometimes it is more useful to use a specific number of elements for training. If this variable is
#       not None but an integer instead, that amount of elements will be chosen for the test set, ignoring
#       the ratio.
TRAIN_NUM: t.Optional[int] = 10
# :param NUM_TARGETS:
#       The size of the vector of ground truth target value annotations for the dataset.
NUM_TARGETS: int = 2
# :param DATASET_TYPE:
#       Currently either "regression" or "classification".
#       This will be used to determine network architectures and visualization options down the line. For
#       example for both types, networks need a different final activation function...
DATASET_TYPE: str = 'classification'

# == STUDENT TEACHER PARAMETERS ==
# These are the parameters which are directly relevant to the process of performing the student teacher
# analysis.
REPETITIONS: int = 25
STUDENT_KWARGS = {
    'units': [5, 5, 5],
    'dropout_rate': 0.0,
    'concat_heads': False,
    'importance_channels': 2,
    'final_units': [5, 2],
    'sparsity_factor': 0.0,
    'use_graph_attributes': False,
    'final_activation': 'linear'
}
HYPER_KWARGS = {
    'epochs': 150,
    'batch_size': 32,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.CategoricalAccuracy(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
}
VARIANT_KWARGS = {
    'ref': {
        'loss_cb': lambda: [ks.losses.CategoricalCrossentropy(from_logits=True),
                            NoLoss(),
                            NoLoss()],
        'loss_weights': [1, 0, 0],
    },
    'exp': {
        'loss_cb': lambda: [ks.losses.CategoricalCrossentropy(from_logits=True),
                            ExplanationLoss(),
                            ExplanationLoss()],
        'loss_weights': [1, 1, 1]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student_megan.py')
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
BASE_PATH = PATH
TESTING = False
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('start_experiment', replace=True)
    def enter_experiment(e):
        e.info('experiment start modifications...')
        if e.p['TESTING']:
            e.info('using testing configuration...')
            e.glob['REPETITIONS'] = 2
            e.glob['HYPER_KWARGS']['epochs'] = 10

    @se.hook('sweep_generator', replace=True)
    def sweep_generator(e, dataset):
        """
        Between each sweep iteration we change the ratio of explanation noise: A specific ratio of
        explanation noise X% means that every importance value for every graph element of every element
        of the dataset has is flipped with a chance of X%.
        """
        NOISE_RATIOS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

        for ratio in NOISE_RATIOS:

            for index, g in enumerate(dataset):

                shape_node = (len(g['node_indices']), 2)
                node_importances_gt = g['node_importances_2'].astype(bool)
                node_importances_random = np.random.randint(2, size=shape_node)
                node_mask = np.random.choice(
                    a=[True, False],
                    p=[ratio, 1-ratio],
                    size=shape_node,
                )
                g['node_importances'] = np.where(
                    node_mask,
                    node_importances_random,
                    node_importances_gt
                ).astype(float)

                shape_edge = (len(g['edge_indices']), 2)
                edge_importances_gt = g['edge_importances_2'].astype(bool)
                edge_importances_random = np.random.randint(2, size=shape_edge)
                edge_mask = np.random.choice(
                    a=[True, False],
                    p=[ratio, 1 - ratio],
                    size=shape_edge,
                )
                g['edge_importances'] = np.where(
                    edge_mask,
                    edge_importances_random,
                    edge_importances_gt
                ).astype(float)

            yield f'noise_{ratio:.2f}'
