import os
import json
import pathlib
import typing as t

import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == DATASET PARAMETERS ==
TEST_INDICES_PATH: str = os.path.join(ASSETS_PATH, 'mutagenicity_exp_test_indices.json')
# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/mutagenicity_exp')
TRAIN_RATIO: float = 0.85
NUM_EXAMPLES: int = 100
EXPLANATION_CHANNELS: int = 2
NUM_TARGETS: int = 1
SEED: int = 1

# == MODEL PARAMETERS ==
UNITS: t.List[int] = [16, 16, 16]
FINAL_UNITS: t.List[int] = [16, 8, 2]
FINAL_ACTIVATION: str = 'softmax'

# == TRAINING PARAMETERS ==
LOSS_CB = lambda: ks.losses.CategoricalCrossentropy()
EPOCHS: int = 15
BATCH_SIZE: int = 25

# == GENERATION PARAMETERS ==
REPETITIONS: int = 3
CONSENSUS_RATIO: float = 0.5

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_create_model_explanations_grad.py')
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
