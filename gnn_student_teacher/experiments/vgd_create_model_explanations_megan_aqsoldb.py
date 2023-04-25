import os
import pathlib
import typing as t

from pycomex.experiment import SubExperiment
from pycomex.util import Skippable

# == DATASET PARAMETERS ==
# The name of the visual graph dataset to use for this experiment.
#VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/aqsoldb')
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/aqsoldb')
TRAIN_RATIO: float = 0.6
NUM_EXAMPLES: int = 100
EXPLANATION_CHANNELS: int = 2
NUM_TARGETS: int = 1
SEED: int = 1

# == MODEL PARAMETERS ==
UNITS: t.List[int] = [30, 30, 30]
DROPOUT_RATE: float = 0.1
CONCAT_HEADS: bool = False
FINAL_UNITS: t.List[int] = [30, 15, 1]
IMPORTANCE_FACTOR: float = 1.0
IMPORTANCE_MULTIPLIER: float = 5.0
SPARSITY_FACTOR: float = 5.0
REGRESSION_REFERENCE: t.Optional[float] = -3
REGRESSION_LIMITS: t.Optional[t.Tuple[float, float]] = [(-12, 4)]

# == GENERATION PARAMETERS ==
REPETITIONS: int = 5
CONSENSUS_RATIO: float = 0.2
POSTFIX: str = 'megan'

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_create_model_explanations_megan.py')
NAMESPACE = 'results/vgd_create_model_explanations_megan_aqsoldb'
BASE_PATH = PATH
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass
