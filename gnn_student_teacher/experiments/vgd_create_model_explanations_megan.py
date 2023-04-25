import os
import pathlib
import typing as t

from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.models import Megan
from graph_attention_student.training import mse, NoLoss

# == MODEL PARAMETERS ==
# All the parameters which determine the behavior of the MEGAN model

UNITS: t.List[int] = [10, 10, 10]
DROPOUT_RATE: float = 0.0
IMPORTANCE_FACTOR: float = 0.0
IMPORTANCE_MULTIPLIER: float = 0.0
SPARSITY_FACTOR: float = 5.0
CONCAT_HEADS: bool = False
FINAL_UNITS: t.List[int] = [10, 5, 1]
FINAL_DROPOUT_RATE: float = 0.0
FINAL_ACTIVATION: str = 'linear'
REGRESSION_REFERENCE: t.Optional[float] = None
REGRESSION_LIMITS: t.Optional[t.Tuple[float, float]] = None

# == GENERATION PARAMETERS ==
REPETITIONS: int = 5
CONSENSUS_RATIO: float = 0.25
POSTFIX: str = 'megan'

# == TRAINING PARAMETERS ==
EPOCHS: int = 100
BATCH_SIZE: int = 64
LOSS_CB = lambda: [mse, NoLoss(), NoLoss()]

# == EVALUATION PARAMETERS ==
VAL_METRIC_KEY: str = 'val_output_1_mean_squared_error'

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_create_model_explanations.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('create_model')
    def create_model(e):
        model = Megan(
            units=e.p['UNITS'],
            dropout_rate=e.p['DROPOUT_RATE'],
            importance_channels=e.p['EXPLANATION_CHANNELS'],
            importance_factor=e.p['IMPORTANCE_FACTOR'],
            importance_multiplier=e.p['IMPORTANCE_MULTIPLIER'],
            sparsity_factor=e.p['SPARSITY_FACTOR'],
            concat_heads=e.p['CONCAT_HEADS'],
            final_units=e.p['FINAL_UNITS'],
            final_dropout_rate=e.p['FINAL_DROPOUT_RATE'],
            final_activation=e.p['FINAL_ACTIVATION'],
            regression_reference=e.p['REGRESSION_REFERENCE'],
            regression_limits=e.p['REGRESSION_LIMITS'],
            use_graph_attributes=False
        )
        return model

    @se.hook('explain', replace=True)
    def explain(e, model, x, y):
        out_pred, ni_pred, ei_pred = model(x)
        return ni_pred, ei_pred




