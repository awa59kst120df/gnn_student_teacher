"""
This string will be saved to the experiment's archive folder as the "experiment description"
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from kgcnn.data.utils import ragged_tensor_from_nested_numpy

from pycomex.experiment import Experiment
from pycomex.util import Skippable

import gnn_student_teacher.typing as tc
from gnn_student_teacher.util import DATASETS_PATH
from gnn_student_teacher.util import resolve_callbacks_nested_dict
from gnn_student_teacher.util import list_to_batches
from gnn_student_teacher.data.persistent import assert_visual_graph_dataset
from gnn_student_teacher.data.persistent import load_visual_graph_dataset
from gnn_student_teacher.data.persistent import update_visual_graph_dataset
from gnn_student_teacher.students import StudentTemplate
from gnn_student_teacher.students.keras import MeganStudent
from gnn_student_teacher.students.keras import KerasStudentTrainer
from gnn_student_teacher.training.losses import NoLoss
from gnn_student_teacher.visualization import plot_node_importances
from gnn_student_teacher.visualization import plot_edge_importances

SHORT_DESCRIPTION = (
    'Will generate MEGAN explanations by running multiple MEGAN semi-supervised trainings'
)

# == DATASET PARAMETERS ==
DATASET_PATH = os.path.join(DATASETS_PATH, 'rb_dual_motifs')
DATASET_NAME = os.path.basename(DATASET_PATH)
METADATA_CONTAINS_INDEX = False
METADATA_CONTAINS_SPLIT = False
METADATA_TARGET_KEY = 'value'
TEST_SPLIT = 0.2
STUDENT_SIZE = 200

# == MODEL PARAMETERS ==
NUM_IMPORTANCE_CHANNELS = 2
STUDENT_KWARGS = {
    'units': [5, 5, 5],
    'final_units': [5, 1],
    'importance_channels': NUM_IMPORTANCE_CHANNELS,
    'sparsity_factor': 0.1,
    'importance_factor': 1.0,
    'importance_multiplier': 5.0,
    'regression_reference': 0.0,
    'regression_limits': (-3, 3)
}

# == TRAINING PARAMETERS ==
REPETITIONS = 5
HYPER_KWARGS = {
    'epochs': 100,
    'batch_size': 64,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
    'loss_cb': lambda: [
        ks.losses.MeanSquaredError(),
        NoLoss(),
        NoLoss()
    ],
    'loss_weights': [1, 0, 0]
}

# == EVALUATION PARAMETERS ==
EVAL_BATCH_SIZE = 1000
EVAL_LOG_STEP = 20
MODIFY_DATASET = True
ANNOTATION_SUFFIX = 'megan'
NORM_ORDER = 0.1

# == EXPERIMENT PARAMETERS ==
NAMESPACE = 'create_explanations/megan_rb_motifs'
DEBUG = True
with Skippable(), (e := Experiment(base_path=os.getcwd(), namespace=NAMESPACE, glob=globals())):
    # "e.info" should be used instead of "print". It will use python's "logging" module to not only
    # print the content ot the console but also write it into a log file at the same time.
    e.info('starting experiment...')
    e.info(f'generating MEGAN explanations for the dataset: "{DATASET_NAME}" @ {DATASET_PATH}')

    e.info(f'loading the dataset...')
    assert_visual_graph_dataset(DATASET_PATH)
    # This "dataset_map" is a mapping of element id / name -> dictionary which contains the information
    # about the element. Note that this dict is NOT the graph dict, but a metadata dict which itself only
    # contains the actual graph dict as one value.
    dataset_map: t.Dict[str, tc.MetaDict] = load_visual_graph_dataset(DATASET_PATH,
                                                                      logger=e.logger, log_step=1000)
    dataset_length = len(dataset_map)
    e.info(f'loaded dataset with {dataset_length} elements')

    e.info('pre-processing dataset for training...')
    indices = list(range(dataset_length))
    # We construct a mapping here of "index within dataset" -> "metadata dictionary".
    dataset_index_map: t.Dict[int, tc.MetaDict] = {}
    dataset: t.List[tc.GraphDict] = [None for _ in indices]
    for name, data in dataset_map.items():
        g: tc.GraphDict = data['metadata']['graph']

        # Some datasets may define separate ground truth for single channel vs multi channel importances.
        # But in this case we definitely need the multi channel definition as the canonical one
        if 'multi_node_importances' in g:
            g['node_importances'] = g['multi_node_importances']
        if 'multi_edge_importances' in g:
            g['edge_importances'] = g['multi_edge_importances']

        target = data['metadata'][METADATA_TARGET_KEY]
        if isinstance(target, list):
            g['graph_labels'] = np.array(target)
        elif isinstance(target, (int, float)):
            g['graph_labels'] = np.array([target])

        # If this flag is true we assume that the canonical index of an element within the dataset is saved
        # as part of its metadata and we will use that index then.
        # Otherwise, we will use the index as the files appeared in the dataset folder (os specific sorting!)
        if METADATA_CONTAINS_INDEX:
            index = int(data['metadata']['index'])
        else:
            index = int(data['index'])

        dataset[index] = g
        dataset_index_map[index] = data

    # ~ Selecting the indices
    # Here we need to create 3 splits of the dataset: First of all we need to declare a test set which
    # will remain unseen throughout all trainings and always just used to assess the quality if explanations.
    # Then we will need a student train set, which will be used for the student teacher analysis later on.
    # This can be relatively small as well (as we know the student teacher analysis works best on small
    # train set sizes) and then the rest of the dataset we will use as the train set for the MEGAN model
    # here.
    # If the "METADATA_CONTAINS_SPLIT" flag is true we will not create this split again but instead use
    # the split that is already contained within the dataset.
    if METADATA_CONTAINS_SPLIT:
        train_indices = [data['metadata']['index']
                         for data in dataset_map.values()
                         if data['metadata']['split'] == 'train']
        test_indices = [data['metadata']['index']
                        for data in dataset_map.values()
                        if data['metadata']['split'] == 'test']
        student_indices = [data['metadata']['index']
                           for data in dataset_map.values()
                           if data['metadata']['split'] == 'student']
    else:
        train_indices = indices
        test_indices = random.sample(train_indices, k=int(len(train_indices) * TEST_SPLIT))
        train_indices = [i for i in train_indices if i not in test_indices]
        student_indices = random.sample(train_indices, k=int(STUDENT_SIZE))
        train_indices = [i for i in train_indices if i not in student_indices]

    e['indices'] = indices
    e['train_indices'] = train_indices
    e['test_indices'] = test_indices
    e['student_indices'] = student_indices
    e.info(f'created dataset split'
           f' - train: {len(train_indices)}'
           f' - test: {len(test_indices)}'
           f' - student: {len(student_indices)}')

    # ~ Setting up the models
    student_template = StudentTemplate(
        student_class=MeganStudent,
        student_name='megan',
        **STUDENT_KWARGS
    )

    # ~ Training the models
    for rep in range(REPETITIONS):
        e.info(f'== REP {rep+1}/{REPETITIONS} ==')

        e.info('instantiation new model from template...')
        model = student_template.instantiate(f'{rep}')
        trainer = KerasStudentTrainer(
            model=model,
            dataset=dataset,
            logger=e.logger,
        )
        result = trainer.fit(**{
            **resolve_callbacks_nested_dict(HYPER_KWARGS),
            'test_indices': test_indices,
            'train_indices': train_indices
        })

        # At the end of the training process we now need to use this model to create explanations /
        # importance tensors for ALL the elements of the dataset and then we need to save them so that we
        # can process them later on...
        e.info('Generating explanations for the entire dataset...')
        index_batches = list_to_batches(indices, batch_size=EVAL_BATCH_SIZE)
        for c, batch in enumerate(index_batches):
            node_attributes = [dataset[i]['node_attributes'] for i in batch]
            node_attributes = ragged_tensor_from_nested_numpy(node_attributes)

            edge_attributes = [dataset[i]['edge_attributes'] for i in batch]
            edge_attributes = ragged_tensor_from_nested_numpy(edge_attributes)

            edge_indices = [dataset[i]['edge_indices'] for i in batch]
            edge_indices = ragged_tensor_from_nested_numpy(edge_indices)

            out, ni, ei = model([node_attributes, edge_attributes, edge_indices])

            for index, out, ni, ei in zip(batch, out.numpy(), ni.numpy(), ei.numpy()):
                e[f'prediction/{rep}/{index}'] = out
                e[f'node_importances/{rep}/{index}'] = ni
                e[f'edge_importances/{rep}/{index}'] = ei

            e.info(f' * processed batch {c}/{len(index_batches)}')

        # ~ Plotting the training results
        e.info('plotting the training results...')
        pdf_path = os.path.join(e.path, f'{rep:02d}_training.pdf')
        with PdfPages(pdf_path) as pdf:
            # Plotting the training loss progression
            epochs = result['epochs']
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
            fig.suptitle('Training Loss')
            ax.plot(result['epochs'], result['history/train/loss/out'],
                    color='blue', ls='--', alpha=0.5, label='train')
            final_value = result['history/train/loss/out'][-1]
            ax.scatter(epochs[-1], final_value, color='blue', alpha=0.5, label=f'{final_value:.2f}')
            ax.plot(result['epochs'], result['history/val/loss/out'],
                    color='blue', ls='-', alpha=1.0, label='test')
            final_value = result['history/val/loss/out'][-1]
            ax.scatter(epochs[-1], final_value, color='blue', alpha=1.0, label=f'{final_value:.2f}')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

            # Plotting the predictions
            y_true = list(result['test/out/true'].values())
            y_pred = list(result['test/out/pred'].values())
            r2 = r2_score(y_true, y_pred)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
            ax.set_xlabel('True Label')
            ax.set_ylabel('Predicted Label')
            fig.suptitle(f'Test Set Scatter Plot\n'
                         f'R2: {r2:.2f}')
            ax.scatter(y_true, y_pred, color='blue', alpha=0.5)
            pdf.savefig(fig)
            plt.close(fig)

            # Plotting the ROC curve for the importances and their ground truth (if it exists)
            fig, (ax_ni, ax_ei) = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
            fig.suptitle('Comparison with Ground truth importances')
            ni_true = [value
                       for index in e['test_indices']
                       for value in np.array(dataset[index]['node_importances']).flatten()]
            ni_pred = [value
                       for index in e['test_indices']
                       for value in np.array(e[f'node_importances/{rep}/{index}']).flatten()]
            fpr, tpr, threshold = roc_curve(ni_true, ni_pred)
            ni_auc = auc(fpr, tpr)
            ax_ni.set_title(f'Node Importances\n'
                            f'AUC: {ni_auc:.3f}')
            ax_ni.plot(fpr, tpr, color='blue')
            ax_ni.set_xlabel('True Positive Rate')
            ax_ni.set_ylabel('False Positive Rate')

            ei_true = [value
                       for index in e['test_indices']
                       for value in np.array(dataset[index]['edge_importances']).flatten()]
            ei_pred = [value
                       for index in e['test_indices']
                       for value in np.array(e[f'edge_importances/{rep}/{index}']).flatten()]
            fpr, tpr, threshold = roc_curve(ni_true, ni_pred)
            ei_auc = auc(fpr, tpr)
            ax_ei.set_title(f'Edge Importances\n'
                            f'AUC: {ei_auc:.3f}')
            ax_ei.plot(fpr, tpr, color='blue')
            ax_ei.set_xlabel('True Positive Rate')
            ax_ei.set_ylabel('False Positive Rate')

            pdf.savefig(fig)
            plt.close(fig)

        e.status()


with Skippable(), e.analysis:
    e.info('starting analysis...')

    e.info('loading dataset...')
    dataset_map: t.Dict[str, tc.MetaDict] = load_visual_graph_dataset(DATASET_PATH,
                                                                      logger=e.logger, log_step=1000)
    dataset_length = len(dataset_map)
    indices = list(range(dataset_length))
    # We construct a mapping here of "index within dataset" -> "metadata dictionary".
    dataset_index_map: t.Dict[int, tc.MetaDict] = {}
    dataset: t.List[tc.GraphDict] = [None for _ in indices]
    for name, data in dataset_map.items():
        g: tc.GraphDict = data['metadata']['graph']

        # Some datasets may define separate ground truth for single channel vs multi channel importances.
        # But in this case we definitely need the multi channel definition as the canonical one
        if 'multi_node_importances' in g:
            g['node_importances'] = g['multi_node_importances']
        if 'multi_edge_importances' in g:
            g['edge_importances'] = g['multi_edge_importances']

        target = data['metadata'][METADATA_TARGET_KEY]
        if isinstance(target, list):
            g['graph_labels'] = np.array(target)
        elif isinstance(target, (int, float)):
            g['graph_labels'] = np.array([target])

        # If this flag is true we assume that the canonical index of an element within the dataset is saved
        # as part of its metadata and we will use that index then.
        # Otherwise, we will use the index as the files appeared in the dataset folder (os specific sorting!)
        if METADATA_CONTAINS_INDEX:
            index = int(data['metadata']['index'])
        else:
            index = int(data['index'])

        dataset[index] = g
        dataset_index_map[index] = data

    # == CALCULATING COMBINED EXPLANATIONS ==
    # Here we calculate the combined explanations, which are essentially an aggregation of all the
    # explanations generated during the individual repetitions of the model training. The idea is that
    # an aggregation should be a bit more stable in regards to noise influences during model training and
    # thus be a bit higher quality than any individual model's explanations
    for index in e['test_indices'] + e['train_indices'] + e['student_indices']:
        nis = []
        eis = []
        for rep in range(REPETITIONS):
            nis.append(e[f'node_importances/{rep}/{index}'])
            eis.append(e[f'edge_importances/{rep}/{index}'])

        # We are going to use this kind of vector norm here instead of just applying the average because
        # this formula will prioritize consensus! So in cases where only one model shows a strong
        # activation and cases where all models show medium activation the latter case will result in a
        # higher value!
        nis = np.array(nis) / _max if (_max := np.max(nis)) != 0 else 1
        eis = np.array(eis) / _max if (_max := np.max(eis)) != 0 else 1
        ni = la.norm(nis, axis=0, ord=NORM_ORDER) / REPETITIONS**(1/NORM_ORDER)
        ei = la.norm(eis, axis=0, ord=NORM_ORDER) / REPETITIONS**(1/NORM_ORDER)
        e[f'node_importances/combined/{index}'] = ni
        e[f'edge_importances/combined/{index}'] = ei

    # == DRAWING THE EXPLANATIONS ==
    # We will use the elements that we defined as "student_indices" also as the example set of elements
    # for which we will visualize all the explanations. specifically we want to visualize the explanations
    # of each individual repetition together with the aggregated explanations to assess their quality.
    e.info('drawing examples...')
    pdf_path = os.path.join(e.path, f'examples.pdf')

    n_rows = int(NUM_IMPORTANCE_CHANNELS)
    n_cols = int(REPETITIONS + 1)
    with PdfPages(pdf_path) as pdf:
        for c, index in enumerate(e['student_indices']):
            g = dataset[index]
            node_coordinates = np.array(g['node_coordinates'])
            data = dataset_index_map[index]

            image = np.asarray(imread(data['image_path']))
            fig, rows = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8 * n_cols, 8 * n_rows),
                                     squeeze=False)
            fig.suptitle(f'Dataset Element: {data["name"]}\n'
                         f'Index: {index}\n')

            # Plotting the aggregated
            col_index = 0
            for row_index in range(n_rows):
                ax = rows[row_index][col_index]
                ax.set_title('Aggregated\n')
                ax.set_ylabel(f'Channel {row_index}')
                ax.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                ni = np.array(e[f'node_importances/combined/{index}'])
                ei = np.array(e[f'edge_importances/combined/{index}'])
                plot_node_importances(
                    g=g,
                    node_importances=ni[:, row_index],
                    node_coordinates=node_coordinates,
                    vmax=np.max(ni),
                    ax=ax
                )
                plot_edge_importances(
                    g=g,
                    edge_importances=ei[:, row_index],
                    node_coordinates=node_coordinates,
                    vmax=np.max(ei),
                    ax=ax
                )

            # Plotting the different variants
            for col_index in range(1, n_cols):
                rep = col_index - 1
                for row_index in range(n_rows):
                    ax = rows[row_index][col_index]
                    ni = np.array(e[f'node_importances/{rep}/{index}'])
                    ei = np.array(e[f'edge_importances/{rep}/{index}'])
                    if row_index == 0:
                        ax.set_title(
                            f'{rep}\n'
                            f'y_true: {g["graph_labels"]} - '
                            f'y_pred: {e[f"prediction/{rep}/{index}"]}'
                        )

                    ax.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    plot_node_importances(
                        g=g,
                        node_importances=ni[:, row_index],
                        node_coordinates=node_coordinates,
                        vmax=np.max(ni),
                        ax=ax
                    )
                    plot_edge_importances(
                        g=g,
                        edge_importances=ei[:, row_index],
                        node_coordinates=node_coordinates,
                        vmax=np.max(ei),
                        ax=ax
                    )

            pdf.savefig(fig)
            plt.close(fig)

            if c % EVAL_LOG_STEP == 0:
                e.info(f' * drawn ({c}/{len(e["student_indices"])})')

    # == MODIFYING THE DATASET ==
    # If the flag "MODIFY_DATASET" is set that means that we should modify the original dataset within the
    # datasets folder by additionally annotating every element in it with the MEGAN generated aggregated
    # explanation!
    if MODIFY_DATASET:
        e.info('annotating the dataset with aggregated explanation...')
        dataset_map: t.Dict[str, tc.MetaDict] = {}
        for index in e['indices']:
            data = dataset_index_map[index]
            metadata = data['metadata']

            metadata['graph'][f'node_importances_{ANNOTATION_SUFFIX}'] = \
                e[f'node_importances/combined/{index}']
            metadata['graph'][f'edge_importances_{ANNOTATION_SUFFIX}'] = \
                e[f'edge_importances/combined/{index}']

            # We also want to save the split information, so the information about whether a particular
            # element belongs to the train, test or student set
            metadata['index'] = index
            if index in e['train_indices']:
                metadata['split'] = 'train'
            elif index in e['test_indices']:
                metadata['split'] = 'test'
            elif index in e['student_indices']:
                metadata['split'] = 'student'

            dataset_map[data['name']] = data

        e.info('updating the persistent dataset records...')
        update_visual_graph_dataset(DATASET_PATH, dataset_map, logger=e.logger, log_step=200)


