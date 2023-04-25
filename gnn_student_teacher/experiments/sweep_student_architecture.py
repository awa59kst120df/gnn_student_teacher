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
from scipy.stats import wilcoxon

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
from gnn_student_teacher.analysis import calculate_student_teacher_simulability
from gnn_student_teacher.training.losses import NoLoss, ExplanationLoss
from gnn_student_teacher.visualization import plot_node_importances, plot_edge_importances
from gnn_student_teacher.visualization import plot_multi_variant_box_plot

matplotlib.use('TkAgg')
np.set_printoptions(precision=2)

SHORT_DESCRIPTION = (
    'Executes a student teacher sweep over different student architectures'
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
    'units': [3, 3, 3],
    'importance_units': [3],
    'importance_channels': NUM_IMPORTANCE_CHANNELS,
    'importance_factor': 0.0,
    'final_units': [3, 1],
    'sparsity_factor': 0.0,
}
UNITS = [3, 3, 3]
FINAL_UNITS = [3, 1]

# == TRAINING PARAMETERS ==
REPETITIONS = 50
TRAINER_CLASS = KerasStudentTrainer
TRAIN_SPLIT = 0.02
TRAIN_SIZE = 100
HYPER_KWARGS = {
    'epochs': 100,
    'batch_size': 16,
    'optimizer_cb': lambda: ks.optimizers.Adam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.MeanSquaredError(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
}
VARIANT_KWARGS = {
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

# == SWEEP PARAMETERS ==
SWEEP = {
    '(2 1) (1)': {
        'UNITS': [2, 1],
        'FINAL_UNITS': [1]
    },
    '(3 2 1) (1)': {
        'UNITS': [3, 2, 1],
        'FINAL_UNITS': [1]
    },
    '(3 3 3) (1)': {
        'UNITS': [3, 3, 3],
        'FINAL_UNITS': [1]
    },
    '(3 3 3) (3 1)': {
        'UNITS': [3, 3, 3],
        'FINAL_UNITS': [3, 1]
    },
    '(5 5 5) (5 1)': {
        'UNITS': [5, 5, 5],
        'FINAL_UNITS': [5, 1]
    },
}

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
NAMESPACE = 'sweep/student_architecture'
DEBUG = False
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
        student_name=f'megan',
        **STUDENT_KWARGS
    )

    student_teacher_analysis = StudentTeacherExplanationAnalysis(
        student_template=student_template,
        trainer_class=KerasStudentTrainer,
        results_class=AbstractStudentResult,
        logger=e.logger
    )

    # ~ Repeated student teacher analysis

    for sweep_name, sweep_params in SWEEP.items():

        # The way a sweep works is that for each element int the sweep dict we specify how the global
        # variables (uppercase) should be overwritten for each iteration of the sweep. We can actually
        # dynamically change the values of variables by using the globals() dict.
        # We then verify that the value has actually changed by dynamically eval-ing that variable.
        e.info(f'==| SWEEP: {sweep_name} |==')
        for param_name, param_value in sweep_params.items():
            globals()[param_name] = param_value
            e.info(f' * {param_name}: {eval(param_name)}')

        # For every element of the sweep we create its own folder within the overall experiment archive
        # folder and all the artifacts will be placed in there. This just makes it more manageable since
        # there will be a lot of artifacts in the end
        sweep_path = os.path.join(e.path, f'{sweep_name}')
        os.mkdir(sweep_path)

        for rep in range(REPETITIONS):

            e.info(f'==| REP ({rep + 1}/{REPETITIONS}) |==')

            # In every repetition we use the same test indices, but depending on the separately chosen
            # TRAIN_SPLIT, we use a different, random subset of the train dataset for the training.
            # (This is important because the student teacher effect only appears for small dataset sizes)
            current_train_indices = random.sample(train_indices, k=TRAIN_SIZE)
            e.info(f'identified {len(current_train_indices)} random elements for train set')
            e[f'train_indices/{sweep_name}/{rep}'] = current_train_indices

            for variant, kwargs in VARIANT_KWARGS.items():
                kwargs['model_kwargs'] = {
                    'units': UNITS,
                    'final_units': FINAL_UNITS
                }

            with tf.device('gpu:0'):
                results = student_teacher_analysis.fit(
                    dataset=dataset,
                    hyper_kwargs={
                        **HYPER_KWARGS,
                        'test_indices': test_indices,
                        'train_indices': current_train_indices
                    },
                    variant_kwargs=VARIANT_KWARGS,
                    suffix=f'{sweep_name.replace(" ","").replace("(", "").replace(")", "")}_{rep}'
                )
                e[f'results/{sweep_name}/{rep}'] = results

            # ~ Processing the results
            # For each repetition we create a separate folder to store the artifacts in, because that will
            # make the archive folder more readable
            folder_path = os.path.join(sweep_path, f'{rep:02d}')
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
                    del model

        e.status()


with Skippable(), e.analysis:
    e.info('starting analysis...')

    # == STUDENT TEACHER BOX PLOT ==
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
    })

    e.info('creating student teacher boxplot')
    data_list = []
    sweep_names = list(SWEEP.keys())
    for sweep in sweep_names:
        data = defaultdict(list)
        for rep, results in e[f'results/{sweep}'].items():
            if any(result['history']['val']['metric']['out'][-1] > 2.9 for result in results.values()):
                # continue
                pass

            for variant, result in results.items():
                value = result['history']['val']['metric']['out'][-1]
                if True or value < 3.0:
                    data[variant].append(value)

        # We want to sort the entries essentially by the variant name here to make sure that later in the
        # plot the boxes appear in the same order for every sweep element.
        data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
        data_list.append(data)

    # Creating the boxplot
    e.info('plotting boxes...')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
    ax.set_title(f'Student Teacher Analysis\n'
                 f'({REPETITIONS} Repetitions)')
    centers = plot_multi_variant_box_plot(
        ax=ax,
        data_list=data_list,
        variant_colors=VARIANT_COLORS
    )

    # Here we will add some lines into the background of the plot which will help to detect a possible
    # trend in the medians of the distributions.
    for variant in VARIANTS:
        plt.plot(centers, [np.median(data[variant]) for data in data_list],
                 color=VARIANT_COLORS[variant], ls='--', zorder=-2, alpha=0.5)

    # Here we calculate the "simulability" metrics for the explanation and reference distributions as well
    # as a wilcoxon non-parametric test for statistical significant differences between the distributions
    if 'ref' in VARIANTS and 'exp' in VARIANTS:
        value_min = min(v for data in data_list for value_list in data.values() for v in value_list)
        value_max = max(v for data in data_list for value_list in data.values() for v in value_list)
        value_range = [value_min - 0.1, value_max + 0.1]
        e.info(f'Identified the total value range {value_range} for all distributions')

        e.info(f'Calculating simulability values from "exp" and "ref" distributions...')
        sims = [calculate_student_teacher_simulability(data['exp'], data['ref'],
                                                       metric_type='falling', value_range=value_range)
                for data in data_list]
        e.info(f'Calculating wilcoxon p values for the distributions...')
        ps = [wilcoxon(data['exp'], data['ref']).pvalue for data in data_list]

        e.info(f'Adding simulability and p values to the plot...')
        # So we want to add these values to the plot as text annotations above each of the sweep sections
        # To do this we need to slightly expand the height of the graph from what it currently is, so we
        # don't accidentally draw the text over parts of the actual box plots.
        y_min, y_max = ax.get_ylim()
        y_max_expanded = y_max + abs(y_min - y_max) * 0.1
        ax.set_ylim([y_min, y_max_expanded])
        y_label = y_max + abs(y_min - y_max) * 0.05
        for center, sim, p in zip(centers, sims, ps):
            if p < 0.05:
                string = r'$\underline{' + f'{sim:.2f}' + r'}$'
            else:
                string = r'$' + f'{sim:.2f}' + r'$'
            ax.text(center, y_label, string)

    # Now we can also use the sweep names as the x tick labels
    ax.set_xticklabels(sweep_names)
    ax.set_xlabel('Train Set Size')
    ax.set_ylabel(f'MSE')

    e.commit_fig('boxplot.pdf', fig)


