import os
import math

import numpy as np
import numpy.linalg as la

from gnn_student_teacher.util import get_version
from gnn_student_teacher.util import render_latex
from gnn_student_teacher.util import list_to_batches

from .util import ASSETS_PATH
from .util import LOG


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


def test_render_latex():
    output_path = os.path.join(ASSETS_PATH, 'out.pdf')
    render_latex({'content': '$\pi = 3.141$'}, output_path)
    assert os.path.exists(output_path)


def test_list_to_batches():
    # First test is such that the batch size fits perfectly
    length = 100
    seq = list(range(100))
    batch_size = 10
    batches = list_to_batches(seq, batch_size=batch_size)
    assert len(batches) == math.ceil(length / batch_size)
    for batch in batches:
        assert len(batch) == batch_size

    # Then a test with an odd number
    length = 87
    seq = list(range(length))
    batches = list_to_batches(seq, batch_size=batch_size)
    assert len(batches) == math.ceil(length / batch_size)
    # make sure all values are in there
    assert len([value for batch in batches for value in batch]) == length


def test_averaging_methods():
    # So the basic motivation here is that I want to have some sort of consensus scheme where the second
    # vector is actually results in a higher value than the first one. Essentially I want to prioritize
    # those vectors where all of the predictions are reasonably close
    vector_1 = [1, 0, 0, 0, 0, 0]
    vector_2 = [0.33, 0.34, 0.33, 0, 0]
    vector_3 = [0.2, 0.2, 0.2, 0.2, 0.2]

    # And a simple averaging does not do that
    mean_1 = np.mean(vector_1)
    mean_2 = np.mean(vector_2)
    mean_3 = np.mean(vector_3)
    LOG.info(f'mean - bad: {mean_1} - medium: {mean_2} - bad: {mean_3}')

    # I guess one alternative would be to scale down the value with the standard deviation
    value_1 = np.mean(vector_1) - np.std(vector_1)
    value_2 = np.mean(vector_2) - np.std(vector_2)
    value_3 = np.mean(vector_3) - np.std(vector_3)
    LOG.info(f'mean minus std - bad: {value_1} - medium: {value_2} - good: {value_3}')

    # what if I basically take the euclidean norm?
    # -> This is doing it exactly the wrong way around
    value_1 = la.norm(vector_1, ord=2)
    value_2 = la.norm(vector_2, ord=2)
    value_3 = la.norm(vector_3, ord=2)
    LOG.info(f'norm2 - bad: {value_1} - medium: {value_2} - good: {value_3}')

    value_1 = la.norm(vector_1, ord=0.5) / len(vector_1)**2
    value_2 = la.norm(vector_2, ord=0.5) / len(vector_2)**2
    value_3 = la.norm(vector_3, ord=0.5) / len(vector_3)**2
    LOG.info(f'norm2 - bad: {value_1} - medium: {value_2} - good: {value_3}')
