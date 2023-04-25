import os
import pathlib
import typing as t

from pycomex.experiment import SubExperiment
from pycomex.util import Skippable

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/rb_adv_motifs')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = []
EXPLANATION_CHANNELS: int = 2
EXPLANATION_POSTFIX: str = '2'
TRAIN_NUM: int = 96
TEST_SUBSET: float = 0.2
NUM_TARGETS: int = 2
DATASET_TYPE: str = 'classification'

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student_gnes.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass