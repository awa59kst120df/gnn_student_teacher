import os
import pathlib

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.training import NoLoss
from graph_attention_student.training import ExplanationLoss

from gnn_student_teacher.students import StudentTemplate
from gnn_student_teacher.students.keras import MeganStudent

# == STUDENT TEACHER PARAMETERS ==
# These are the parameters which are directly relevant to the process of performing the student teacher
# analysis.
REPETITIONS: int = 3
STUDENT_KWARGS = {
    'units': [5, 5, 5],
    'concat_heads': False,
    'importance_channels': 2,
    'final_units': [2],
    'sparsity_factor': 1,
    'use_graph_attributes': False,
    'final_activation': 'softmax'
}
HYPER_KWARGS = {
    'epochs': 100,
    'batch_size': 8,
    'optimizer_cb': lambda: ks.optimizers.Nadam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.CategoricalAccuracy(),
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
        'loss_weights': [1, 0.1, 0.1]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student.py')
NAMESPACE = 'results/vgd_student_megan'
BASE_PATH = PATH
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    # == PARAMETER HOOKS ==
    # The main thing we need to accomplish with the parameter hooks here is that we need to dynamically
    # change the network configuration depending on whether the current dataset is a regression or a
    # classification dataset!

    @se.hook('STUDENT_KWARGS')
    def student_kwargs(e, value: dict):
        # We need to make sure that the network output matches the dataset
        value['final_units'][-1] = e.p['NUM_TARGETS']

        if e.p['DATASET_TYPE'] == 'classification':
            value['final_activation'] = 'softmax'
        elif e.p['DATASET_TYPE'] == 'regression':
            value['final_activation'] = 'linear'

        return value

    @se.hook('VARIANT_KWARGS')
    def variant_kwargs(e, value: dict):
        # For the regression/classification case we need to use the appropriate loss and metric functions!
        if e.p['DATASET_TYPE'] == 'classification':
            value['ref']['loss_cb'] = lambda: [
                ks.losses.CategoricalCrossentropy(),
                NoLoss(),
                NoLoss(),
            ]
            value['exp']['loss_cb'] = lambda: [
                ks.losses.CategoricalCrossentropy(),
                ExplanationLoss(),
                ExplanationLoss(),
            ]

        elif e.p['DATASET_TYPE'] == 'regression':
            value['ref']['loss_cb'] = lambda: [
                ks.losses.MeanSquaredError(),
                NoLoss(),
                NoLoss(),
            ]
            value['exp']['loss_cb'] = lambda: [
                ks.losses.MeanSquaredError(),
                ExplanationLoss(),
                ExplanationLoss(),
            ]

        return value

    @se.hook('HYPER_KWARGS')
    def hyper_kwargs(e, value: dict):
        if e.p['DATASET_TYPE'] == 'classification':
            value['prediction_metric_cb'] = lambda: ks.metrics.CategoricalAccuracy()

        elif e.p['DATASET_TYPE'] == 'regression':
            value['prediction_metric_cb'] = lambda: ks.metrics.MeanSquaredError()

        return value

    # == CUSTOM HOOKS ==

    @se.hook('create_student_template', default=True)
    def create_student_template(e):
        return StudentTemplate(
            student_class=MeganStudent,
            student_name=f'megan',
            **e.p['STUDENT_KWARGS']
        )
