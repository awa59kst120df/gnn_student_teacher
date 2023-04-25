"""
This experiment performs a student teacher analysis on the "rb_adv_motifs" dataset using a GNES student.

CHANGELOG

0.1.0 - 01.04.2023 - initial version
"""
import os
import pathlib
import random
import json
import typing as t
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.training import LogProgressCallback

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

INDICES_PATH = os.path.join(ASSETS_PATH, 'mutagenicity_exp_test_indices.json')

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/mutagenicity_exp')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = [
    os.path.join(ASSETS_PATH, 'expansion_mutagenicity_exp_2_megan'),
    os.path.join(ASSETS_PATH, 'expansion_mutagenicity_exp_2_grad'),
    os.path.join(ASSETS_PATH, 'expansion_mutagenicity_exp_2_gnnx')
]
HAS_EXPLANATIONS: bool = True
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
    'dropout_rate': 0.1,
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
    'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
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
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    def over_sample_dataset(label_index_map: t.Dict[int, t.List[int]]) -> t.List[int]:
        num_max = max([len(indices) for indices in label_index_map.values()])
        indices_result = []
        for label, indices in label_index_map.items():
            indices_result += indices

            num_diff = num_max - len(indices)
            if num_diff > 10:
                indices_result += random.choices(indices, k=num_diff)

        return indices_result


    @se.hook('sweep_generator', replace=True)
    def sweep_generator(e, dataset):
        with open(INDICES_PATH, mode='r') as file:
            content = file.read()
            indices = json.loads(content)

        for index in indices:
            g = dataset[index]
            g['node_importances'] = g['node_importances_2']
            g['edge_importances'] = g['edge_importances_2']

        e['overwrite_indices'] = indices
        yield 'gt'

        for index in indices:
            g = dataset[index]
            g['node_importances'] = g['node_importances_2_megan']
            g['edge_importances'] = g['edge_importances_2_megan']

        e['overwrite_indices'] = indices
        yield 'megan'

        for index in indices:
            g = dataset[index]
            g['node_importances'] = g['node_importances_2_gnnx']
            g['edge_importances'] = g['edge_importances_2_gnnx']

        e['overwrite_indices'] = indices
        yield 'gnnx'

        for index in indices:
            g = dataset[index]
            g['node_importances'] = g['node_importances_2_grad']
            g['edge_importances'] = g['edge_importances_2_grad']

        e['overwrite_indices'] = indices
        yield 'grad'

        # The last baseline is completely random explanations. These are only in as a "sanity check" so to
        # say. For these explanations there should not exist any effect at all!
        for index in indices:
            g = dataset[index]

            node_importances = np.array(g['node_importances_2']).copy()
            edge_importances = np.array(g['edge_importances_2']).copy()

            node_importances = np.random.choice([0, 1], size=node_importances.shape, p=[0.5, 0.5])
            edge_importances = np.random.choice([0, 1], size=edge_importances.shape, p=[0.5, 0.5])

            g['node_importances'] = node_importances
            g['edge_importances'] = edge_importances

        e['overwrite_indices'] = indices
        yield 'random'
