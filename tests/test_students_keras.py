import typing as t

import pytest
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np

from gnn_student_teacher.training.losses import NoLoss
from gnn_student_teacher.data.structures import AbstractStudentResult
from gnn_student_teacher.students import AbstractStudentModel
from gnn_student_teacher.students.keras import MockKerasStudent
from gnn_student_teacher.students.keras import KerasStudentTrainer
from gnn_student_teacher.students.keras import MeganStudent

from .util import generate_mock_dataset_keras
from .util import LOG


# === FIXTURES ===

@pytest.fixture
def mock_keras_student() -> MockKerasStudent:
    student = MockKerasStudent(
        'mock', 'ref',
        hidden_units=5,
        importance_units=3,
        output_units=1
    )
    return student


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


# === UNITTESTS ===

class TestMockKerasStudent:

    def test_mock_keras_student_basically_works(self):
        hidden_units = 10
        output_units = 2
        importance_units = 3
        student = MockKerasStudent(
            'mock', 'ref',
            hidden_units=hidden_units,
            output_units=output_units,
            importance_units=importance_units,
        )
        assert isinstance(student, MockKerasStudent)
        assert isinstance(student, AbstractStudentModel)
        assert isinstance(student, ks.models.Model)

        # Doing it the direct way using __call__
        node_input = tf.ragged.constant([
            [[1, 1], [1, 1], [1, 1]],
            [[2, 2], [2, 2]]
        ], ragged_rank=1, dtype=tf.float32)
        edge_index = tf.ragged.constant([
            [[0, 1], [1, 2]],
            [[0, 1]]
        ], ragged_rank=1, dtype=tf.int32)
        edge_input = tf.ragged.constant([
            [[1], [1]],
            [[2]]
        ], ragged_rank=1, dtype=tf.float32)
        out, ni, ei = student([node_input, edge_input, edge_index])
        assert out.shape == (2, output_units)
        assert ni.shape == (2, None, importance_units)
        assert ei.shape == (2, None, importance_units)

    def test_predict_single(self):
        hidden_units = 10
        output_units = 2
        importance_units = 3
        student = MockKerasStudent(
            'mock', 'ref',
            hidden_units=hidden_units,
            output_units=output_units,
            importance_units=importance_units,
        )

        # Using predict_single
        out, ni, ei = student.predict_single(
            node_attributes=np.array([[1, 1], [1, 1], [1, 1]], dtype=float),
            edge_attributes=np.array([[0], [1]], dtype=float),
            edge_indices=np.array([[0, 1], [1, 2]], dtype=int)
        )

        assert isinstance(out, np.ndarray)
        assert out.shape == (output_units, )
        assert isinstance(ni, np.ndarray)
        assert ni.shape == (3, importance_units)
        assert isinstance(ei, np.ndarray)
        assert ei.shape == (2, importance_units)


class TestKerasStudentTrainer:

    def test_construction_works(self):
        hidden_units = 10
        output_units = 2
        importance_units = 3
        student = MockKerasStudent(
            'mock', 'ref',
            hidden_units=hidden_units,
            output_units=output_units,
            importance_units=importance_units
        )
        dataset = generate_mock_dataset_keras(
            num_samples=10,
            num_outputs=1,
            num_node_attributes=3,
            num_edge_attributes=3,
            num_importance_channels=3,
        )
        trainer = KerasStudentTrainer(student, dataset)
        assert isinstance(trainer, KerasStudentTrainer)

    def test_training_process_works(self, mock_keras_student, keras_dataset):
        manager = KerasStudentTrainer(
            model=mock_keras_student,
            dataset=keras_dataset,
            logger=LOG
        )
        result = manager.fit(
            optimizer=ks.optimizers.Adam(learning_rate=0.01),
            loss=[ks.losses.MeanSquaredError(), NoLoss(), NoLoss()],
            loss_weights=[1, 0, 0],
            prediction_metric=ks.metrics.MeanSquaredError(),
            importance_metric=ks.metrics.MeanAbsoluteError(),
            epochs=5,
            log_progress=1,
        )
        assert isinstance(result, AbstractStudentResult)
        assert len(result['epochs']) == 5
        assert isinstance(result['history/train/loss/out'], list)
        assert result['history/train/loss/out'][-1] < result['history/train/loss/out'][0]
        assert 0.0 < result['test/auroc/ni'] < 1.0
        assert 0.0 < result['test/auroc/ei'] < 1.0

class TestMeganStudent:

    def test_construction_basically_works(self):
        student = MeganStudent(
            name='megan',
            variant='ref',
            units=[10, 10],
            importance_factor=0.0,
            importance_channels=3,
            return_importances=True,
        )
        assert isinstance(student, MeganStudent)
        assert isinstance(student, AbstractStudentModel)

    def test_training_basically_works(self, keras_dataset):
        student = MeganStudent(
            name='megan',
            variant='ref',
            units=[10, 10],
            importance_factor=0.0,
            importance_channels=3,
        )
        trainer = KerasStudentTrainer(
            model=student,
            dataset=keras_dataset,
            logger=LOG
        )
        result = trainer.fit(
            optimizer=ks.optimizers.Adam(),
            loss=[ks.losses.MeanSquaredError(), NoLoss(), NoLoss()],
            loss_weights=[1, 0, 0],
            prediction_metric=ks.metrics.MeanSquaredError(),
            importance_metric=ks.metrics.MeanAbsoluteError(),
            epochs=5
        )
        assert result['history/train/loss/out'][0] > result['history/train/loss/out'][-1]
