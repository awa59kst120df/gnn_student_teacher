import os
import pathlib

from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.conv.gcn_conv import GCN
from graph_attention_student.models.gradient import grad_importances

PATH = pathlib.Path(__file__).parent.absolute()

REPETITIONS = 25

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_student_gnes_rb_adv_motifs_2.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('sweep_generator')
    def sweep_generator(e, dataset):
        e.parameters['STUDENT_KWARGS'] = {
            'batch_size': e.p['BATCH_SIZE'],
            'units': [8, 8, 8],
            'final_units': [8, 2],
            'final_activation': 'softmax',
            'importance_func': lambda *args, **kwargs: grad_importances(*args, **kwargs, use_relu=True),
            'layer_cb': lambda k: GCN(units=k)
        }
        yield 'gcn'

        e.parameters['STUDENT_KWARGS'] = {
            'batch_size': e.p['BATCH_SIZE'],
            'units': [5, 5, 5],
            'final_units': [5, 2],
            'final_activation': 'softmax',
            'importance_func': lambda *args, **kwargs: grad_importances(*args, **kwargs, use_relu=True),
            'layer_cb': lambda k: AttentionHeadGATV2(units=k, use_edge_features=True)
        }
        yield 'gatv2'

