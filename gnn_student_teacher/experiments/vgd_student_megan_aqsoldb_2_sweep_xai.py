import os
import pathlib
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
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/aqsoldb')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = [
    os.path.join(PATH, 'assets/expansion_aqsoldb_2_megan')
]
HAS_EXPLANATIONS: bool = False
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
TRAIN_NUM: t.Optional[int] = 150
# :param NUM_TARGETS:
#       The size of the vector of ground truth target value annotations for the dataset.
NUM_TARGETS: int = 1
# :param DATASET_TYPE:
#       Currently either "regression" or "classification".
#       This will be used to determine network architectures and visualization options down the line. For
#       example for both types, networks need a different final activation function...
DATASET_TYPE: str = 'regression'

# == STUDENT TEACHER PARAMETERS ==
# These are the parameters which are directly relevant to the process of performing the student teacher
# analysis.
REPETITIONS: int = 25
STUDENT_KWARGS = {
    'units': [5, 5, 5],
    'concat_heads': False,
    'importance_channels': 2,
    'dropout_rate': 0.1,  # 0.05
    'final_units': [5, 1],
    'final_dropout_rate': 0,
    'sparsity_factor': 0.0,
    'use_graph_attributes': False,
    'final_activation': 'linear',
    # 'importance_transformations': [
    #     CoefficientActivation('relu', -1.0),
    #     CoefficientActivation('relu', +1.0)
    # ]
}
HYPER_KWARGS = {
    'epochs': 100,
    'batch_size': 16,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
}
VARIANT_KWARGS = {
    'ref': {
        'loss_cb': lambda: [ks.losses.MeanSquaredError(),
                            NoLoss(),
                            NoLoss()],
        'loss_weights': [1, 0, 0],
    },
    'exp': {
        'loss_cb': lambda: [ks.losses.MeanSquaredError(),
                            ExplanationLoss(),
                            ExplanationLoss()],
        'loss_weights': [1, 1, 1]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student_megan.py')
NAMESPACE = 'results/vgd_student_megan_aqsoldb_2_sweep_xai'
BASE_PATH = PATH
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('sweep_generator', replace=True)
    def sweep_generator(e, dataset):
        # Another interesting comparison is not completely random explanations but somewhat the most
        # "trivial" explanations. In this case I want to highlight only carbon atoms in one channel and
        # only oxygen atoms in the other channel.
        # The hope would be that this shows a small effect, as it is actually not too far removed from
        # how solubility works, but the effect should obviously still be smaller than with the
        # completely correct explanations
        overwrite_indices = []
        for index, g in enumerate(dataset):
            if 'node_importances_2_megan' in g and 'edge_importances_2_megan' in g:
                # We set the node importances for the first channel for all the nodes which are carbon
                # atoms and for the second channel for all the nodes which are oxygen atoms.
                node_importances = np.zeros(shape=(len(g['node_indices']), 2))
                node_importances[g['node_attributes'][:, 1] == 1, 0] = 1
                cond = np.concatenate(
                    [
                        np.expand_dims(g['node_attributes'][:, 2] == 1, axis=-1),
                        np.expand_dims(g['node_attributes'][:, 3] == 1, axis=-1),
                    ],
                    axis=-1
                )
                node_importances[cond.any(axis=-1), 1] = 1

                g['node_importances'] = node_importances

                # As edge importances we simply set it such that all the edges which are adjacent to one
                # "important" node are also marked as important.
                edge_importances = np.zeros(shape=(len(g['edge_indices']), 2))
                for m, (i, j) in enumerate(g['edge_indices']):
                    # For carbon we only want to mark edges between the atoms.
                    if node_importances[i, 0] == 1 and node_importances[j, 0] == 1:
                        edge_importances[m, 0] = 1
                    # For the others we want all adjacent edges.
                    if node_importances[i, 1] == 1 or node_importances[j, 1] == 1:
                        edge_importances[m, 1] = 1

                g['edge_importances'] = edge_importances

                overwrite_indices.append(index)

        e['overwrite_indices'] = overwrite_indices
        # yield 'trivial'

        # Then obviously we want to use the explanations which were generated by a larger MEGAN model
        overwrite_indices = []
        for index, g in enumerate(dataset):
            if 'node_importances_2_megan' in g and 'edge_importances_2_megan' in g:
                g['node_importances'] = g['node_importances_2_megan']
                g['edge_importances'] = g['edge_importances_2_megan']
                overwrite_indices.append(index)

        e['overwrite_indices'] = overwrite_indices
        yield 'megan'

        # The last baseline is completely random explanations. These are only in as a "sanity check" so to
        # say. For these explanations there should not exist any effect at all!
        overwrite_indices = []
        for index, g in enumerate(dataset):
            if 'node_importances_2_megan' in g and 'edge_importances_2_megan' in g:
                node_importances = np.array(g['node_importances_2_megan']).copy()
                edge_importances = np.array(g['edge_importances_2_megan']).copy()

                node_importances = np.random.choice([0, 1], size=node_importances.shape, p=[0.5, 0.5])
                edge_importances = np.random.choice([0, 1], size=edge_importances.shape, p=[0.5, 0.5])

                g['node_importances'] = node_importances
                g['edge_importances'] = edge_importances

                overwrite_indices.append(index)

        e['overwrite_indices'] = overwrite_indices
        yield 'random'

