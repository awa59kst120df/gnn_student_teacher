import os
import json
import time
import logging
import typing as t
from collections import defaultdict

import numpy as np

import gnn_student_teacher.typing as tc
from gnn_student_teacher.util import NULL_LOGGER


def assert_visual_graph_dataset(path: str) -> None:
    """
    Given an absolute file system path, this method will run assertions to make sure that that path is
    indeed a "visual dataset" format. If that is not the case, an AssertionError will be raised.

    :returns: None
    """
    assert os.path.isdir(path), (
        f'The given dataset path "{path}" does not represent a valid visual graph dataset because it is not '
        f'a directory. A visual graph dataset must be a directory where each element is represented by two '
        f'files.'
    )

    files = os.listdir(path)
    assert len(files) != 0, (
        f'The given dataset path "{path}" is an empty directory! Thus not a valid visual graph dataset.'
        f'Make sure to provide the correct folder path to a non-empty folder'
    )
    assert len(files) % 2 == 0, (
        f'The given dataset path "{path}" contains an uneven number of elements. For a valid visual graph '
        f'dataset, each element of the dataset has to be represented by exactly two files. The uneven '
        f'number of files indicates a violation of this assumption.'
    )

    counter = defaultdict(int)
    for file in files:
        try:
            file_name, file_extension = file.split('.')
            if file_extension in ['png', 'json']:
                counter[file_name] += 1

        except ValueError:
            raise AssertionError(f'The dataset index of the file "{file}" from the dataset "{path}" could '
                                 f'not be reconstructed! Not a valid visual graph dataset')

    faulty_indices = [i for i, c in counter.items() if c != 2]
    assert len(faulty_indices) == 0, (
        f'Some elements in the dataset "{path}" are not represented by two files, as it is required for a '
        f'valid visual graph dataset: {faulty_indices}'
    )


def load_visual_graph_dataset(path: str,
                              logger: logging.Logger = NULL_LOGGER,
                              log_step: int = 100,
                              ) -> t.Dict[str, dict]:
    dataset_map = defaultdict(dict)

    files = os.listdir(path)
    num_files = len(files)

    logger.info(f'loading visual graph dataset with {num_files} files...')
    start_time = time.time()
    for c, file in enumerate(files):
        name, extension = file.split('.')
        file_path = os.path.join(path, file)

        dataset_map[name]['name'] = name

        if extension in ['png']:
            dataset_map[name]['image_path'] = file_path

        if extension in ['json']:
            dataset_map[name]['metadata_path'] = file_path
            with open(file_path, mode='r') as json_file:
                metadata = json.loads(json_file.read())

                dataset_map[name]['metadata'] = metadata

        if c % log_step == 0:
            logger.info(f' * ({c}/{num_files})'
                        f' - processed file: {file}'
                        f' - elapsed time: {time.time() - start_time:.1f}s')

    for c, (name, data) in enumerate(dataset_map.items()):
        data['index'] = c

    return dict(dataset_map)


def update_visual_graph_dataset(path: str,
                                dataset_map: t.Dict[str, tc.MetaDict],
                                logger: logging.Logger = NULL_LOGGER,
                                log_step: int = 100,
                                indent: t.Optional[int] = None):
    assert_visual_graph_dataset(path)

    dataset_length = len(dataset_map)
    for c, (name, data) in enumerate(dataset_map.items()):
        json_path = os.path.join(path, f'{name}.json')

        prev_stat = os.stat(json_path)
        with open(json_path, mode='w') as json_file:
            content = json.dumps(data['metadata'], indent=indent, cls=NumericJsonEncoder)
            json_file.write(content)

        new_stat = os.stat(json_path)

        if c % log_step == 0:
            logger.info(f' * updated ({c}/{dataset_length})'
                        f' - name: {name}'
                        f' - size: {prev_stat.st_size:.1f}B -> {new_stat.st_size:.1f}B')


class NumericJsonEncoder(json.JSONEncoder):

    def default(self, o: t.Any) -> t.Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super(NumericJsonEncoder, self).default(o)
