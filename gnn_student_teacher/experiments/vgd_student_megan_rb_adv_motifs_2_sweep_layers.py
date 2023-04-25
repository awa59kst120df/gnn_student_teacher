"""
This experiment conducts a student teacher analysis for the synthetic rb adversarial motifs dataset, where
the layer structure of the MEGAN student is being changed.

CHANGELOG

0.1.0 - 03.04.2023 - Initial version
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
DATASET_FILTER_KEY: t.Optional[str] = None  # model_explanations
TRAIN_NUM: t.Optional[int] = 100
NUM_TARGETS: int = 2
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
    'batch_size': 16,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.001),
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
        In this sweep the architecture / layer structure of the student model is changed between each of
        the iterations generally going from very few parameters to more parameters and layers.d
        """
        e.glob['STUDENT_KWARGS'].update({
            'units': [3, 3],
            'dropout_rate': 0.0,
            'final_units': [3, 2],
        })
        yield '3 3 | 3 2'

        e.glob['STUDENT_KWARGS'].update({
            'units': [3, 3, 3],
            'dropout_rate': 0.0,
            'final_units': [3, 2],
        })
        yield '3 3 3 | 3 2'

        e.glob['STUDENT_KWARGS'].update({
            'units': [5, 5, 5],
            'dropout_rate': 0.0,
            'final_units': [5, 2],
        })
        yield '5 5 5 | 5 2'

        e.glob['STUDENT_KWARGS'].update({
            'units': [10, 10, 10],
            'dropout_rate': 0.0,
            'final_units': [10, 2],
        })
        yield '10 10 10 | 10 2'

        e.glob['STUDENT_KWARGS'].update({
            'units': [20, 20, 20],
            'dropout_rate': 0.0,
            'final_units': [20, 2],
        })
        yield '20 20 20 | 20 2'

