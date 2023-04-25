"""
This experiment performs a student teacher analysis on the "rb_adv_motifs" dataset using a GNES student.

CHANGELOG

0.1.0 - 01.04.2023 - initial version
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
from graph_attention_student.training import LogProgressCallback

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/bbbp')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = []
HAS_EXPLANATIONS: bool = False
EXPLANATION_CHANNELS: int = 2
EXPLANATION_POSTFIX: str = '1'
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
TRAIN_NUM: t.Optional[int] = 100
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
REPETITIONS: int = 35
STUDENT_KWARGS = {
    'units': [5, 5, 5],
    'dropout_rate': 0.1,
    'concat_heads': False,
    'importance_channels': 2,
    'final_units': [5, 2],
    'sparsity_factor': 0.0,
    'use_graph_attributes': False,
    'final_activation': 'linear'
}
HYPER_KWARGS = {
    'epochs': 200,
    'batch_size': 32,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
}
VARIANT_KWARGS = {
    'ref': {
        'loss_cb': lambda: [ks.losses.CategoricalCrossentropy(),
                            NoLoss(),
                            NoLoss()],
        'loss_weights': [1, 0, 0],
    },
    'exp': {
        'loss_cb': lambda: [ks.losses.CategoricalCrossentropy(),
                            ExplanationLoss(),
                            ExplanationLoss()],
        'loss_weights': [1, 1, 1]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())


# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student_megan.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('sweep_generator', replace=True)
    def sweep_generator(e, dataset):
        # The last baseline is completely random explanations. These are only in as a "sanity check" so to
        # say. For these explanations there should not exist any effect at all!
        overwrite_indices = []
        label_index_map = {0: [], 1: []}
        for index, g in enumerate(dataset):
            if 'node_importances_2_gao' in g and 'edge_importances_2_gao' in g:
                label = int(np.argmax(g['graph_labels']))
                label_index_map[label].append(index)

                node_importances = np.array(g['node_importances_2_gao']).copy()
                edge_importances = np.array(g['edge_importances_2_gao']).copy()

                node_importances = np.random.choice([0, 1], size=node_importances.shape, p=[0.5, 0.5])
                edge_importances = np.random.choice([0, 1], size=edge_importances.shape, p=[0.5, 0.5])

                g['node_importances'] = node_importances
                g['edge_importances'] = edge_importances

                overwrite_indices.append(index)

        e['overwrite_indices'] = overwrite_indices + random.choices(label_index_map[0], k=100)
        yield 'random'

        # The real question is if the human annotations show an effect.
        overwrite_indices = []
        label_index_map = {0: [], 1: []}
        for index, g in enumerate(dataset):
            if 'node_importances_2_gao' in g and 'edge_importances_2_gao' in g:
                label = int(np.argmax(g['graph_labels']))
                label_index_map[label].append(index)

                # We set the node importances for the first channel for all the nodes which are carbon
                # atoms and for the second channel for all the nodes which are oxygen atoms.
                node_importances = g['node_importances_2_gao']
                g['node_importances'] = node_importances

                # As edge importances we simply set it such that all the edges which are adjacent to one
                # "important" node are also marked as important.
                edge_importances = g['edge_importances_2_gao']
                g['edge_importances'] = edge_importances

                overwrite_indices.append(index)

        e['overwrite_indices'] = overwrite_indices + random.choices(label_index_map[0], k=100)
        yield 'gao_human'
