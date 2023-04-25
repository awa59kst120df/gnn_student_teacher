import copy
import typing as t

import pytest

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from gnn_student_teacher.students import StudentTemplate
from gnn_student_teacher.students import AbstractStudentModel
from gnn_student_teacher.students import AbstractStudentTrainer
from gnn_student_teacher.students.keras import MockKerasStudent
from gnn_student_teacher.students.keras import KerasStudentTrainer
from gnn_student_teacher.data.structures import AbstractStudentResult
from gnn_student_teacher.training.losses import NoLoss, ExplanationLoss
from gnn_student_teacher.analysis import StudentTeacherExplanationAnalysis

from .util import generate_mock_dataset_keras


@pytest.fixture()
def mock_keras_student_template() -> StudentTemplate:
    template = StudentTemplate(
        student_class=MockKerasStudent,
        student_name='keras_mock',
        hidden_units=10,
        output_units=1,
        importance_units=3
    )
    return template


@pytest.fixture
def keras_dataset() -> t.List[dict]:
    dataset = generate_mock_dataset_keras(
        num_samples=32,
        num_outputs=1,
        num_node_attributes=10,
        num_edge_attributes=1,
        num_importance_channels=3,
    )
    return dataset


# == StudentTeacherExplanationAnalysis ==

class TestStudentTeacherExplanationAnalysis:

    def test_construction_basically_works(self, mock_keras_student_template):
        student_teacher = StudentTeacherExplanationAnalysis(
            student_template=mock_keras_student_template,
            trainer_class=KerasStudentTrainer,
            results_class=AbstractStudentResult
        )
        assert isinstance(student_teacher, StudentTeacherExplanationAnalysis)

    def test_merge_kwargs_works(self):
        """
        ``merge_kwargs`` is a class method which is supposed to take a list of kwarg dicts and merge them
        into a single list, where the dicts further at the end of the list can overwrite fields
        """
        # First of all the thing which should definitely work is if there are just completely exclusive
        # kwargs dicts
        kwargs_list = [
            {'value1': 1.0},
            {'value2': 2.0},
            {'value3': 3.0}
        ]
        merged = StudentTeacherExplanationAnalysis._merge_kwargs(kwargs_list)
        assert isinstance(merged, dict)
        assert 'value1' in merged
        assert 'value2' in merged
        assert 'value3' in merged

        # Now we test if the overwriting works
        kwargs_list = [
            {'metric': 'mae', 'loss': 'mae'},
            {'loss': 'mse', 'optimizer': 'adam'}
        ]
        merged = StudentTeacherExplanationAnalysis._merge_kwargs(kwargs_list)
        assert merged['metric'] == 'mae'
        assert merged['loss'] == 'mse'
        assert merged['optimizer'] == 'adam'

        # Another feature of this method is that it should automatically evaluate all callbacks. So every
        # element of the dict whose name ends with "_cb" is interpreted as a callable to be evaluated
        kwargs_list = [
            {'metric_cb': lambda: 'mae', 'loss': 'mae'},
            {'loss_cb': lambda: 'mse', 'optimizer': 'adam'}
        ]
        merged = StudentTeacherExplanationAnalysis._merge_kwargs(kwargs_list)
        print(merged)
        assert 'loss' in merged
        assert 'loss_cb' not in merged
        assert merged['loss'] == 'mse'
        assert merged['metric'] == 'mae'

    def test_mock_keras_student_analysis_works(self,
                                               mock_keras_student_template,
                                               keras_dataset):
        """
        Executes a student teacher analysis using the keras mock student and a mock dataset
        """
        student_teacher = StudentTeacherExplanationAnalysis(
            student_template=mock_keras_student_template,
            trainer_class=KerasStudentTrainer,
            results_class=AbstractStudentResult
        )
        results = student_teacher.fit(
            dataset=keras_dataset,
            hyper_kwargs={
                'epochs': 5,
                'batch_size': 16,
                'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
                'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
                'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
                'log_progress': 1,
            },
            variant_kwargs={
                'ref': {
                    'loss_weights': [1, 0, 0],
                    'loss': [ks.losses.MeanSquaredError(), NoLoss(), NoLoss()]
                },
                'exp': {
                    'loss_weights': [1, 1, 1],
                    'loss': [ks.losses.MeanSquaredError(), ExplanationLoss(), ExplanationLoss()]
                }
            }
        )
        assert isinstance(results, dict)
        assert 'ref' in results
        assert 'exp' in results
        assert results['ref']['history']['train']['loss']['ni'][0] == 0
        assert results['exp']['history']['train']['loss']['ni'][0] != 0




