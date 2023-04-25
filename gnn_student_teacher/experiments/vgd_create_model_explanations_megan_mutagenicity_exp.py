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

from graph_attention_student.training import NoLoss

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == DATASET PARAMETERS ==
TEST_INDICES_PATH: str = os.path.join(ASSETS_PATH, 'mutagenicity_exp_test_indices.json')
# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/mutagenicity_exp')
# VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/aqsoldb')
TRAIN_RATIO: float = 0.6
NUM_EXAMPLES: int = 100
EXPLANATION_CHANNELS: int = 2
NUM_TARGETS: int = 2
SEED: int = 1

# == MODEL PARAMETERS ==
UNITS: t.List[int] = [30, 30, 30]
DROPOUT_RATE: float = 0.1
CONCAT_HEADS: bool = False
FINAL_UNITS: t.List[int] = [30, 15, 2]
IMPORTANCE_FACTOR: float = 1.0
IMPORTANCE_MULTIPLIER: float = 0.4
SPARSITY_FACTOR: float = 2.0
FINAL_ACTIVATION: str = 'softmax'
REGRESSION_REFERENCE: t.Optional[float] = None
REGRESSION_LIMITS: t.Optional[t.Tuple[float, float]] = None

# == TRAINING PARAMETERS ==
EPOCHS = 25
OPTIMIZER_CB: t.Callable[[], ks.optimizers.Optimizer] = lambda: ks.optimizers.Adam(learning_rate=0.001)

# == GENERATION PARAMETERS ==
REPETITIONS: int = 3
CONSENSUS_RATIO: float = 0.5
POSTFIX: str = 'megan'

VAL_METRIC_KEY = 'output_1_categorical_accuracy'

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_create_model_explanations_megan.py')
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
BASE_PATH = PATH
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('overwrite_test_indices')
    def overwrite_test_indices(e, dataset, test_indices):

        with open(TEST_INDICES_PATH, mode='r') as file:
            content = file.read()
            indices = json.loads(content)

        e.info(f'loaded canonical test indices from file: {TEST_INDICES_PATH}')
        e.info(f'loaded {len(indices)} test indices')

        return indices

    def over_sample_dataset(label_index_map: t.Dict[int, t.List[int]]) -> t.List[int]:
        num_max = max([len(indices) for indices in label_index_map.values()])
        indices_result = []
        for label, indices in label_index_map.items():
            indices_result += indices

            num_diff = num_max - len(indices)
            if num_diff > 10:
                indices_result += random.choices(indices, k=num_diff)

        return indices_result

    @se.hook('overwrite_train_indices')
    def overwrite_train_indices(e, dataset, train_indices):
        e.info('Over-sampling the training dataset to achieve equal class distribution!')
        label_index_map = defaultdict(list)
        for index in train_indices:
            g = dataset[index]
            label = int(np.argmax(g['graph_labels']))
            label_index_map[label].append(index)

        return over_sample_dataset(label_index_map)

    @se.hook('compile_model', default=True)
    def compile_model(e, model):
        model.compile(
            optimizer=e.p['OPTIMIZER_CB'](),
            loss=[
                ks.losses.CategoricalCrossentropy(),
                NoLoss(),
                NoLoss(),
            ],
            metrics=[ks.metrics.CategoricalAccuracy()],
            run_eagerly=e.p['RUN_EAGERLY'],
        )
        return model
