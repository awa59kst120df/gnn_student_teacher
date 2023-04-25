import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.layers import CoefficientActivation

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/rb_dual_motifs')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = [
    # os.path.join(PATH, 'assets/expansion_rb_dual_motifs_megan')
]
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
TRAIN_NUM: t.Optional[int] = 100
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
    'final_units': [5, 1],
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
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student_megan.py')
NAMESPACE = 'results/vgd_student_megan_rb_motifs_2'
BASE_PATH = PATH
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass
