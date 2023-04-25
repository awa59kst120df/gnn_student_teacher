import os

import pytest

from gnn_student_teacher.data.persistent import assert_visual_dataset
from gnn_student_teacher.data.persistent import load_visual_graph_dataset

from .util import LOG
from .util import ASSETS_PATH


def test_assert_visual_graph_dataset():
    # In the basic case we expect an error if we pass the path of a single file which is obviously wrong
    dataset_path = os.path.join(ASSETS_PATH, 'README.md')
    with pytest.raises(AssertionError):
        assert_visual_dataset(dataset_path)

    # In the test assets we have two folders, one which we know is in fact a valid visual dataset and
    # the other which is faulty.
    dataset_path = os.path.join(ASSETS_PATH, 'mock_visual_graph_dataset_valid')
    assert_visual_dataset(dataset_path)

    dataset_path = os.path.join(ASSETS_PATH, 'mock_visual_graph_dataset_invalid')
    with pytest.raises(AssertionError):
        assert_visual_dataset(dataset_path)


def test_load_visual_graph_dataset():
    dataset_path = os.path.join(ASSETS_PATH, 'mock_visual_graph_dataset_valid')
    dataset = load_visual_graph_dataset(dataset_path, logger=LOG, log_step=1)
    assert isinstance(dataset, dict)
    # We know that this dataset has 3 elements (aka 6 files)
    assert len(dataset) == 3
    assert '000' in dataset
    assert 'image_path' in dataset['000'] and 'metadata_path' in dataset['000']
    assert os.path.exists(dataset['000']['image_path'])
    assert os.path.exists(dataset['000']['metadata_path'])
    # metadata should have been properly loaded from the json file
    assert 'metadata' in dataset['000']
    assert len(dataset['000']['metadata']) != 0