import os
import pathlib
import random
import shutil
import traceback
import json
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.conv.gcn_conv import GCN
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.visualization.importances import create_importances_pdf
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.training import mse
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.util import array_normalize
from gnn_student_teacher.util import clean_binary_edge_importances

# == DATASET PARAMETERS ==
# These parameters define the dataset that is to be used for the experiment as well as properties of
# that dataset such as the train test split for example.

# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/rb_dual_motifs')
TRAIN_RATIO: float = 0.6
NUM_EXAMPLES: int = 100
EXPLANATION_CHANNELS: int = 1
NUM_TARGETS: int = 1
SEED: int = 1

# == GENERATION PARAMETERS ==
REPETITIONS: int = 3
CONSENSUS_RATIO: float = 0.3
POSTFIX: str = 'model'

# == EVALUATION PARAMETERS ==
VAL_METRIC_KEY: str = 'val_mean_squared_error'
RENDER_INDIVIDUAL_EXPLANATIONS: bool = True

# == MODEL PARAMETERS ==
# These parameters are used for the model

class MockModel(ks.models.Model):

    def __init__(self,
                 units: int = 32,
                 num_outputs: int = 1,
                 ):
        super(MockModel, self).__init__()
        self.conv_layers = [
            GCN(units=units, use_bias=True),
            GCN(units=units, use_bias=True),
            GCN(units=units, use_bias=True)
        ]
        self.lay_pooling = PoolingNodes(pooling_method='sum')
        self.lay_final = DenseEmbedding(units=num_outputs)

    def call(self, inputs, training=False):
        node_input, edge_input, edge_index_input = inputs
        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_index_input])
        pooled = self.lay_pooling(x)
        out = self.lay_final(pooled)
        return out


# == TRAINING PARAMETERS ==
OPTIMIZER_CB: t.Callable[[], ks.optimizers.Optimizer] = lambda: ks.optimizers.Adam(learning_rate=0.001)
LOSS_CB: t.Callable[[], t.Callable] = lambda: mse
DEVICE: str = 'cpu:0'
EPOCHS: int = 10
BATCH_SIZE: int = 32
RUN_EAGERLY: bool = False

# == EVALUATION PARAMETERS ==
LOG_STEP_EVAL: int = 1000

# == EXPERIMENT PARAMETERS ==
PATH = os.getcwd()
BASE_PATH = PATH
NAMESPACE = 'results/vgd_create_model_explanations'
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    e.info('creating explanations for VGD using MEGAN model...')

    e.info('loading visual graph dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )
    dataset_size = len(index_data_map)

    dataset: t.List[dict] = [None for _ in range(dataset_size)]
    dataset_indices: t.List[int] = []
    for index, data in index_data_map.items():
        g = data['metadata']['graph']
        g['node_importances'] = np.zeros(shape=(len(g['node_indices']), EXPLANATION_CHANNELS))
        g['edge_importances'] = np.zeros(shape=(len(g['edge_indices']), EXPLANATION_CHANNELS))

        dataset[index] = g
        dataset_indices.append(index)

    # We need those later
    dataset_size = len(dataset)
    dataset_indices_set = set(dataset_indices)

    # First of all we want to create a new dataset split for this new repetition, which we then
    # also need to process into the correct format of ragged tensors which can be used to train the
    # model.
    # This specific implementation using sets may seem like unnecessary hassle, but it is A LOT
    # faster for bigger datasets!
    e.info('creating train-test split...')
    random.seed(SEED)

    test_indices = random.sample(dataset_indices, k=int(dataset_size * (1 - TRAIN_RATIO)))

    # :hook overwrite_test_indices:
    #       This filer hook can optionally be used to use a specific set of test indices instead of just
    #       a random sample, for example if a canonical train test split exists for the dataset.
    test_indices = e.apply_hook(
        'overwrite_test_indices',
        default=test_indices,
        dataset=dataset,
        test_indices=test_indices,
    )

    test_indices_set = set(test_indices)
    train_indices = [index for index in dataset_indices if index not in test_indices_set]

    # :hook overwrite_train_indices:
    #       This filter hook can optionally be used to overwrite the list of train indices. This may
    #       primarily be interesting for classification datasets, where it is necessary to over-sample
    #       certain classes to ensure a balanced training dataset.
    train_indices = e.apply_hook(
        'overwrite_train_indices',
        dataset=dataset,
        train_indices=train_indices,
        default=train_indices
    )

    e[f'train_indices'] = train_indices
    e[f'test_indices'] = test_indices
    e.info(f'identified {len(train_indices)} train and {len(test_indices)} test')
    e.info(f'first few train indices: {train_indices[:10]}')
    e.info(f'first few test indices: {test_indices[:10]}')

    num_examples = min(NUM_EXAMPLES, len(test_indices))
    example_indices = random.sample(test_indices, k=num_examples)
    e['example_indices'] = example_indices

    x_train, y_train, x_test, y_test = process_graph_dataset(
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    @e.hook('create_model', default=True)
    def create_model(_e):
        """
        This default implementation will return a mock model
        """
        model = MockModel(
            num_outputs=_e.p['NUM_TARGETS']
        )
        return model

    @e.hook('compile_model', default=True)
    def compile_model(_e, model):
        model.compile(
            optimizer=_e.p['OPTIMIZER_CB'](),
            loss=_e.p['LOSS_CB'](),
            metrics=[mse],
            run_eagerly=_e.p['RUN_EAGERLY'],
        )
        return model

    @e.hook('fit_model', default=True)
    def fit_model(_e, model, x_train, y_train, x_test, y_test):
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=_e.p['EPOCHS'],
            batch_size=_e.p['BATCH_SIZE'],
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=0,
            callbacks=[
                LogProgressCallback(
                    logger=_e.logger,
                    identifier=_e.p['VAL_METRIC_KEY'],
                    epoch_step=5,
                )
            ]
        )
        return history.history

    @e.hook('explain')
    def explain(_e, model, x, y):
        node_input, edge_input, edge_index_input = x

        node_importances = tf.ones_like(node_input)
        node_importances = tf.reduce_mean(node_importances, axis=-1, keepdims=True)

        edge_importances = tf.ones_like(edge_input)
        edge_importances = tf.reduce_mean(edge_importances, axis=-1, keepdims=True)

        return node_importances, edge_importances


    with tf.device(DEVICE):

        for rep in range(REPETITIONS):
            e.info(f'REP ({rep+1}/{REPETITIONS})')

            # :hook create_model:
            #   This hook has to return a model which should be able to train on the given task as well as
            #   to generate explanations for each prediction! This function also has the responsibility to
            #   compile the model
            model: ks.models.Model = e.apply_hook('create_model')
            # :hook compile_model:
            #   This hook ...
            e.apply_hook('compile_model', model=model)

            e.info('fitting model...')
            hist: dict = e.apply_hook(
                'fit_model',
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            e.info('model training finished')
            e.info(f'for model with {model.count_params()} parameters')

            # After the model training is done we need to create the explanations for the test dataset and
            # then save those to the experiment so that in the end we can aggregate all the different
            # explanations of the different repetitions into one single explanation per element.

            e.info('creating the explanations...')
            node_importances, edge_importances = e.apply_hook(
                'explain',
                model=model,
                x=x_test,
                y=y_test
            )
            node_importances = node_importances.numpy()
            edge_importances = edge_importances.numpy()

            for c, index in enumerate(test_indices):
                ni = array_normalize(node_importances[c])
                ei = array_normalize(edge_importances[c])
                e[f'ni/{index}/{rep}'] = ni
                e[f'ei/{index}/{rep}'] = ei

            e.info('saved explanations for current repetition')


with Skippable(), e.analysis:
    e.info('starting analysis...')
    e.info('loading dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )

    @e.hook('aggregate_importances', default=True)
    def aggregate_importances(_e, values):
        values_sum = np.sum(values, axis=0)
        threshold = _e.p['REPETITIONS'] * _e.p['CONSENSUS_RATIO']
        return (values_sum > threshold).astype(int)

    e.info('aggregating explanations...')
    for index in e['test_indices']:
        g = index_data_map[index]['metadata']['graph']

        ni_values = np.array([e[f'ni/{index}/{rep}'] for rep in range(REPETITIONS)])
        ni_agg = e.apply_hook('aggregate_importances', values=ni_values)
        e[f'ni/{index}/agg'] = ni_agg

        ei_values = np.array([e[f'ei/{index}/{rep}'] for rep in range(REPETITIONS)])
        ei_agg = e.apply_hook('aggregate_importances', values=ei_values)

        # 21.04.2023 - A big problem with all the explanations which are not MEGAN are that with a lot
        # of them they are not connected graphs which means that there may just be individual node
        # and edge explanations somewhere in the graph. That will cause problems, especially the edge
        # explanations, so we use this utility function here to clean up all the edge explanations
        # which are not connected to any graph...
        edge_importances = clean_binary_edge_importances(ei_agg, g['edge_indices'], ni_agg)

        e[f'ei/{index}/agg'] = ei_agg

    # Since we have modified the experiment data in this analysis we also want to save these additions which
    # we have made persistently to the JSON file.
    e.info('saving experiment data...')
    e.save_experiment_data()

    e.info('visualizing examples...')
    graph_list = [index_data_map[i]['metadata']['graph'] for i in e['example_indices']]
    image_path_list = [index_data_map[i]['image_path'] for i in e['example_indices']]
    node_positions_list = [g['node_positions'] for g in graph_list]
    labels_list = [f'Element "{i}"' for i in e['example_indices']]
    importances_map = {
        'agg': (
            [e[f'ni/{i}/agg'] for i in e['example_indices']],
            [e[f'ei/{i}/agg'] for i in e['example_indices']]
        ),
    }
    if RENDER_INDIVIDUAL_EXPLANATIONS:
        for rep in range(REPETITIONS):
            importances_map[str(rep)] = (
                [e[f'ni/{i}/{rep}'] for i in e['example_indices']],
                [e[f'ei/{i}/{rep}'] for i in e['example_indices']]
            )
    pdf_path = os.path.join(e.path, 'examples.pdf')
    create_importances_pdf(
        graph_list=graph_list,
        image_path_list=image_path_list,
        node_positions_list=node_positions_list,
        importances_map=importances_map,
        labels_list=labels_list,
        output_path=pdf_path,
        show_x_ticks=True,
        show_y_ticks=True,
        logger=e.logger,
        log_step=10,
    )

    # ~ Persistently saving the generated explanations
    # Here we create what is called a VGD expansion for the current VGD dataset. This is simply a folder
    # with json files, which each contain additional metadata annotations for a subset of the dataset
    # elements. such an expansion can be very easily loaded and added to a VGD using existing function
    # from the library. This will make it easier to work with those generated explanations.

    e.info('saving the aggregated explanations as VGD expansion...')
    expansion_path = os.path.join(e.path, 'expansion')
    if os.path.exists(expansion_path):
        shutil.rmtree(expansion_path)
    os.mkdir(expansion_path)

    for index in e['test_indices']:
        file_name = f'{index:02d}.json'
        file_path = os.path.join(expansion_path, file_name)

        node_importances = e[f'ni/{index}/agg']
        edge_importances = e[f'ei/{index}/agg']

        metadata = {
            # This is an absolute must to even be able to load the expansion later on
            'index': index,
            # This is a flag which we can use later on to easily identify which of the elements are
            # annotated with such a generated explanation.
            'model_explanations': 1,
            # Here we add the generated explanations as additional attributes of the graph data structure
            'graph': {
                f'node_importances_{EXPLANATION_CHANNELS}_{POSTFIX}': node_importances,
                f'edge_importances_{EXPLANATION_CHANNELS}_{POSTFIX}': edge_importances,
            }
        }

        with open(file_path, mode='w') as file:
            content = json.dumps(metadata, cls=NumericJsonEncoder)
            file.write(content)

    file_names = os.listdir(expansion_path)
    num_files = len(file_names)
    e.info(f'created VGD expansion with {num_files} elements @ {expansion_path}')

