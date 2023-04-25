"""
The base experiment which uses the GNES student to conduct the student teacher analysis.

CHANGELOG

0.1.0 - 01.04.2023 - initial version
"""
import os
import pathlib

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.training import mae as mae_loss
from graph_attention_student.training import mse as mse_loss
from graph_attention_student.models.gradient import grad_importances

from gnn_student_teacher.students import StudentTemplate
from gnn_student_teacher.students.keras import GnesStudent

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
TRAIN_NUM: int = 16

# == TRAINING PARAMETERS ==
REPETITIONS: int = 3
BATCH_SIZE: int = 16

# == STUDENT PARAMETERS ==
STUDENT_KWARGS = {
    'batch_size': BATCH_SIZE,
    'units': [5, 5, 5],
    'final_units': [5, 2],
    'final_activation': 'softmax',
    'importance_func': lambda *args, **kwargs: grad_importances(*args, **kwargs, use_relu=True)
}
HYPER_KWARGS = {
    'epochs': 150,
    'batch_size': BATCH_SIZE,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.CategoricalAccuracy(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
    # The GNES student does not support the execution of the "test_step" during the keras fit() method!
    # A final evaluation on the test step will still be calculated though.
    'do_eval': False,
    'test_batch_size': 32,
}
VARIANT_KWARGS = {
    'ref': {
        'loss_cb': lambda: [ks.losses.CategoricalCrossentropy(),
                            NoLoss(),
                            NoLoss()],
        'loss_weights': [1, 0, 0],
    },
    'exp': {
        'loss_cb': lambda: [ks.losses.CategoricalCrossentropy(),
                            ExplanationLoss(loss_function=mse_loss),
                            ExplanationLoss(loss_function=mse_loss)],
        'loss_weights': [1, 2, 2]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())

# == EVALUATION PARAMETERS ==

# Since the GNES student does not compute metrics during the training process, it also does not make sense
# to plot the training process.
PLOT_TRAINING = False

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('create_student_template')
    def create_student_template(e):
        e.info('creating GNES student template...')
        print(e.p['STUDENT_KWARGS'])
        return StudentTemplate(
            student_class=GnesStudent,
            student_name=f'gnes',
            **e.p['STUDENT_KWARGS']
        )
