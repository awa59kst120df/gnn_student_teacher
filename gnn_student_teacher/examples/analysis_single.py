"""
This string will be saved to the experiment's archive folder as the "experiment description"
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import typing as t
from pprint import pprint
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread

from pycomex.experiment import Experiment
from pycomex.util import Skippable

from gnn_student_teacher.util import DATASETS_PATH
from gnn_student_teacher.data.structures import AbstractStudentResult
from gnn_student_teacher.data.persistent import assert_visual_dataset
from gnn_student_teacher.data.persistent import load_visual_graph_dataset
from gnn_student_teacher.students import StudentTemplate
from gnn_student_teacher.students.keras import KerasStudentTrainer
from gnn_student_teacher.students.keras import MeganStudent
from gnn_student_teacher.analysis import StudentTeacherExplanationAnalysis
from gnn_student_teacher.training.losses import NoLoss, ExplanationLoss
from gnn_student_teacher.visualization import plot_node_importances, plot_edge_importances
from gnn_student_teacher.visualization import plot_multi_variant_box_plot

matplotlib.use('TkAgg')
np.set_printoptions(precision=2)

SHORT_DESCRIPTION = (
    'An example executing a single student teacher analysis for a given dataset'
)

# == DATASET PARAMETERS ==
DATASET_PATH = os.path.join(DATASETS_PATH, 'rb_dual_motifs')
METADATA_CONTAINS_INDEX = False
METADATA_TARGET_KEY = 'value'
IS_REGRESSION = True
TEST_SPLIT = 0.2

# == STUDENT MODEL PARAMETERS ==
STUDENT_CLASS = MeganStudent
NUM_IMPORTANCE_CHANNELS = 2
STUDENT_KWARGS = {
    'units': [3, 2, 1],
    'importance_units': [],
    'importance_channels': NUM_IMPORTANCE_CHANNELS,
    'importance_factor': 0.0,
    'final_units': [1],
    'sparsity_factor': 0.0,
}

# == TRAINING PARAMETERS ==
REPETITIONS = 5
TRAINER_CLASS = KerasStudentTrainer
TRAIN_SPLIT = 0.025
HYPER_KWARGS = {
    'epochs': 100,
    'batch_size': 16,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
}
VARIANT_KWARGS = {
    # 'reg': {
    #     'loss_cb': lambda: [ks.losses.MeanSquaredError(),
    #                         NoLoss(),
    #                         NoLoss()],
    #     'loss_weights': [1, 0, 0],
    #     'model_kwargs': {
    #         'importance_factor': 1.0,
    #         'importance_multiplier': 1.0,
    #         'regression_reference': 0,
    #         'regression_limits': (-3, +3),
    #     }
    # },
    'ref': {
        'loss_cb': lambda: [ks.losses.MeanSquaredError(),
                            NoLoss(),
                            NoLoss()],
        'loss_weights': [1, 0, 0],
    },
    'exp': {
        'loss_cb': lambda: [ks.losses.MeanSquaredError(),
                            ExplanationLoss(),
                            ExplanationLoss()],
        'loss_weights': [1, 1, 1]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())

# == EVALUATION PARAMETERS ==
LOG_STEP = 10
NUM_EXAMPLES = 20
VARIANT_COLORS = {
    'reg': 'red',
    'ref': 'blue',
    'exp': 'green',
}
SAVE_MODELS = False

# == EXPERIMENT PARAMETERS ==
NAMESPACE = 'analysis_single'
DEBUG = True
with Skippable(), (e := Experiment(base_path=os.getcwd(), namespace=NAMESPACE, glob=globals())):

    # ~ Loading the dataset
    e.info(f'Loading the dataset "{DATASET_PATH}" ...')
    e.info(
        'The dataset has to be in the "visual graph dataset" format. That implies that the dataset should '
        'be specified as a path to a folder, which contains 2 files for each element of the dataset: A '
        'PNG file which contains a visualization of the graph and a JSON file which contains meta data '
        'as well as the graph representation of the element'
    )
    assert_visual_dataset(DATASET_PATH)
    dataset_map: t.Dict[str, dict] = load_visual_graph_dataset(DATASET_PATH)
    dataset_length = len(dataset_map)
    e.info(f'loaded dataset with {dataset_length} elements')

    e.info(f'Converting dataset to appropriate format...')
    # the "visual graph dataset" format as we have loaded it will make things a bit better
    indices = list(range(dataset_length))
    dataset_index_map: t.Dict[int, dict] = {}
    dataset: t.List[dict] = [None for _ in indices]
    for name, data in dataset_map.items():
        g = data['metadata']['graph']

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

    # ~ Setting up the test dataset
    # The test dataset is independent of whatever happens in the individual repetitions. We will use the
    # same test dataset for all the variants across all repetitions. This will increase the comparability
    # of the results we are getting
    test_indices = random.sample(indices, k=int(TEST_SPLIT * dataset_length))
    train_indices = [i for i in indices if i not in test_indices]
    e['test_indices'] = test_indices
    e.info(f'identified test set with {len(test_indices)} elements, which will be used for all repetitions')
    # We will also use the same examples across all repetitions
    num_examples = min(NUM_EXAMPLES, len(test_indices))
    example_indices = random.sample(indices, k=num_examples)
    e['example_indices'] = example_indices
    e.info(f'from the test set, identified {len(example_indices)} example elements')

    # ~ Setting up the analysis
    e.info(f'Using student template with student class: {STUDENT_CLASS.__name__}')
    student_template = StudentTemplate(
        student_class=STUDENT_CLASS,
        student_name='student',
        **STUDENT_KWARGS
    )

    student_teacher_analysis = StudentTeacherExplanationAnalysis(
        student_template=student_template,
        trainer_class=KerasStudentTrainer,
        results_class=AbstractStudentResult,
        logger=e.logger
    )

    # ~ Repeated student teacher analysis

    for rep in range(REPETITIONS):

        e.info(f'==| REP ({rep + 1}/{REPETITIONS}) |==')

        # In every repetition we use the same test indices, but depending on the separately chosen
        # TRAIN_SPLIT, we use a different, random subset of the train dataset for the training.
        # (This is important because the student teacher effect only appears for small dataset sizes)
        current_train_indices = random.sample(train_indices, k=int(len(train_indices) * TRAIN_SPLIT))
        e.info(f'identified {len(current_train_indices)} random elements for train set')
        e[f'train_indices/{rep}'] = current_train_indices

        results = student_teacher_analysis.fit(
            dataset=dataset,
            hyper_kwargs={
                **HYPER_KWARGS,
                'test_indices': test_indices,
                'train_indices': current_train_indices
            },
            variant_kwargs=VARIANT_KWARGS,
        )
        e[f'results/{rep}'] = results

        # ~ Processing the results
        folder_path = os.path.join(e.path, f'{rep:02d}')
        os.mkdir(folder_path)

        # Visualizing the individual repetition results
        pdf_path = os.path.join(folder_path, f'training.pdf')
        with PdfPages(pdf_path) as pdf:
            # Plotting the loss curves
            for variant in VARIANTS:
                for curve_type in ['loss', 'metric']:
                    fig, (ax_out, ax_ni, ax_ei) = plt.subplots(ncols=3, nrows=1, figsize=(8 * 3, 8))
                    fig.suptitle(f'{curve_type.upper()} during Training: {variant}')

                    ax_out.set_title('Prediction Output')
                    ax_out.plot(results[variant]['epochs'],
                                results[variant]['history']['train'][curve_type]['out'],
                                color=VARIANT_COLORS[variant], alpha=0.7, ls='-.', label='train')
                    ax_out.plot(results[variant]['epochs'],
                                results[variant]['history']['val'][curve_type]['out'],
                                color=VARIANT_COLORS[variant], alpha=1.0, ls='-', label='val')
                    final_value = results[variant]['history']['val'][curve_type]['out'][-1]
                    ax_out.scatter(results[variant]['epochs'][-1], final_value,
                                   color=VARIANT_COLORS[variant], alpha=1.0,
                                   label=f'{final_value:.2f}')
                    ax_out.legend()

                    ax_ni.set_title('Node Importances')
                    ax_ni.plot(results[variant]['epochs'],
                               results[variant]['history']['train'][curve_type]['ni'],
                               color=VARIANT_COLORS[variant], alpha=0.7, ls='-.', label='train')
                    ax_ni.plot(results[variant]['epochs'],
                               results[variant]['history']['val'][curve_type]['ni'],
                               color=VARIANT_COLORS[variant], alpha=1.0, ls='-', label='val')
                    ax_ni.legend()

                    ax_ei.set_title('Edge Importances')
                    ax_ei.plot(results[variant]['epochs'],
                               results[variant]['history']['train'][curve_type]['ei'],
                               color=VARIANT_COLORS[variant], alpha=0.7, ls='-.', label='train')
                    ax_ei.plot(results[variant]['epochs'],
                               results[variant]['history']['val'][curve_type]['ei'],
                               color=VARIANT_COLORS[variant], alpha=1.0, ls='-', label='val')
                    ax_ei.legend()

                    pdf.savefig(fig)
                    plt.close(fig)

                if IS_REGRESSION:
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
                    ax.set_ylabel('Predicted Value')
                    ax.set_xlabel('True Value')
                    for index in test_indices:
                        y_true = results[variant]['test']['out']['true'][str(index)]
                        y_pred = results[variant]['test']['out']['pred'][str(index)]
                        ax.scatter(y_true, y_pred, alpha=0.5, color=VARIANT_COLORS[variant])

                    y_lim_lower, y_lim_upper = ax.get_ylim()
                    x_lim_lower, x_lim_upper = ax.get_xlim()
                    lim_lower = min(y_lim_lower, x_lim_lower)
                    lim_upper = max(y_lim_upper, x_lim_upper)
                    ax.set_ylim([lim_lower, lim_upper])
                    ax.set_xlim([lim_lower, lim_upper])
                    ax.plot([lim_lower, lim_upper], [lim_lower, lim_upper], color='black', alpha=0.5)

                    pdf.savefig(fig)
                    plt.close(fig)

        # Drawing examples from the dataset together with their explanations
        e.info('drawing examples...')
        pdf_path = os.path.join(folder_path, f'examples.pdf')

        n_rows = int(NUM_IMPORTANCE_CHANNELS)
        n_cols = int(len(VARIANT_KWARGS) + 1)
        with PdfPages(pdf_path) as pdf:
            for c, index in enumerate(example_indices):
                g = dataset[index]
                node_coordinates = np.array(g['node_coordinates'])
                data = dataset_index_map[index]

                image = np.asarray(imread(data['image_path']))
                fig, rows = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8 * n_cols, 8 * n_rows),
                                         squeeze=False)
                fig.suptitle(f'Dataset Element: {data["name"]}\n'
                             f'Index: {index}\n')

                # Plotting the ground truth
                col_index = 0
                for row_index in range(n_rows):
                    ax = rows[row_index][col_index]
                    ax.set_title('Ground Truth\n')
                    ax.set_ylabel(f'Channel {row_index}')
                    ax.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    plot_node_importances(
                        g=g,
                        node_importances=np.array(g['node_importances'])[:, row_index],
                        node_coordinates=node_coordinates,
                        ax=ax
                    )
                    plot_edge_importances(
                        g=g,
                        edge_importances=np.array(g['edge_importances'])[:, row_index],
                        node_coordinates=node_coordinates,
                        ax=ax
                    )

                # Plotting the different variants
                for col_index in range(1, n_cols):
                    variant = VARIANTS[col_index - 1]
                    out, ni, ei = results[variant]['model'].predict_single(
                        g['node_attributes'],
                        g['edge_attributes'],
                        g['edge_indices']
                    )
                    for row_index in range(n_rows):
                        ax = rows[row_index][col_index]
                        if row_index == 0:
                            ax.set_title(
                                f'{variant}\n'
                                f'y_true: {g["graph_labels"]} - '
                                f'y_pred: {out}'
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

        # Saving the model
        for variant, result in results.items():
            model = result['model']
            # We need to delete this entry from the results dict here because later on this result dict
            # needs to be JSON serialized and the model object will not allow that.
            del result['model']

            if SAVE_MODELS:
                # Instead we explicitly serialize the model here
                model_path = os.path.join(folder_path, f'{variant}.model')
                model.save(model_path)


with Skippable(), e.analysis:
    e.info('starting analysis...')

    # Student Teacher Box Plot
    data = defaultdict(list)
    for rep, results in e['results'].items():
        for variant, result in results.items():
            value = result['history']['val']['metric']['out'][-1]
            data[variant].append(value)

    data_list = [data]

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.set_title(f'Student Teacher Analysis\n'
                 f'({REPETITIONS} Repetitions)')
    ax.set_ylabel(f'MSE')
    plot_multi_variant_box_plot(
        ax=ax,
        data_list=data_list,
        variant_colors=VARIANT_COLORS
    )

    e.commit_fig('boxplot.pdf', fig)


