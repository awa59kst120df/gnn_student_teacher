"""
Base experiment to perform a student teacher experiments based on a visual graph dataset (VGD).

This base experiment realizes a generic implementation which can be configured in subsequent SubExperiment
extensions. For example, it supports both classification and regression tasks.

CHANGELOG

0.1.0 - 21.02.2023 - Initial version

0.2.0 - 28.03.2023 - (1) Added the "start_experiment" hook at the very start which can be used to implement
a TESTING version of the experiment. (2) In the analysis now also creating a latex table with the results
as well as various other small fixes.

0.3.0 - 29.03.2023 - Now some elements of each student teacher analysis results dict are being deleted
before that dict is committed to the main experiment storage, because there were some serious memory issues
recently.

0.4.0 - 01.04.2023 - Some necessary changes to support the GNES student. (1) The plotting of the training
progress can now be turned off, because that is not supported for the GNES student. (2) It is now possible
to use only a subset of the test set for the student teacher evaluation. (3) Moved the creation of the
student template into the sweep loop.
"""
import os
import pathlib
import random
import typing as t
from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_rel
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import load_visual_graph_dataset_expansion
from graph_attention_student.util import latex_table
from graph_attention_student.util import render_latex
from graph_attention_student.util import latex_table_element_mean

from gnn_student_teacher.analysis import StudentTemplate
from gnn_student_teacher.analysis import AbstractStudentResult
from gnn_student_teacher.analysis import StudentTeacherExplanationAnalysis
from gnn_student_teacher.analysis import cha_srihari_distance
from gnn_student_teacher.analysis import calculate_sts
from gnn_student_teacher.students.keras import KerasStudentTrainer
from gnn_student_teacher.students.keras import MeganStudent
from gnn_student_teacher.training.losses import NoLoss, ExplanationLoss
from gnn_student_teacher.visualization import plot_multi_variant_box_plot

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/mock')
VISUAL_GRAPH_DATASET_EXPANSION_PATHS: t.List[str] = [
    # os.path.join(PATH, 'assets/expansion_rb_dual_motifs_megan')
]
# :param HAS_EXPLANATIONS:
#       This flag determines if the base dataset even has any explanation annotations. This may not be
#       the case for new real-world datasets with no known ground truth explanations! If this is set to
#       False, empty explanations will be used as the target explanations for the analysis
HAS_EXPLANATIONS: bool = True
EXPLANATION_CHANNELS: int = 2
EXPLANATION_POSTFIX: str = '2'
# :param DATASET_SELECT_KEY:
#       Optionally this may specify a string metadata key for the VGD. The corresponding value for each
#       element of the dataset is supposed to be a boolean value. If that value is True then the item
#       will be considered part of the dataset, otherwise it won't. This is essentially a means to
#       pre-filter the dataset so that only a subset is used for the student teacher analysis.
DATASET_FILTER_KEY: t.Optional[str] = None  # model_explanations
# :param TRAIN_RATIO:
#       The ratio of the dataset to be used for the training of the students. The rest of it will
#       implicitly be used for the testing
TRAIN_RATIO: float = 0.8
# :param TRAIN_NUM:
#       Sometimes it is more useful to use a specific number of elements for training. If this variable is
#       not None but an integer instead, that amount of elements will be chosen for the test set, ignoring
#       the ratio.
TRAIN_NUM: t.Optional[int] = 50
# :param TEST_SUBSET:
#       This may optionally be a float value in the range between 0 and 1 which defines the ratio of the
#       actual test set to be used to evaluate the model. Some models may not be able to handle the large
#       test sets that often result from the student teacher process efficiently. Thus, they may only use
#       a still sufficiently large subset of the test set for the evaluation.
TEST_SUBSET: t.Optional[float] = None
# :param NUM_TARGETS:
#       The size of the vector of ground truth target value annotations for the dataset.
NUM_TARGETS: int = 2
# :param DATASET_TYPE:
#       Currently either "regression" or "classification".
#       This will be used to determine network architectures and visualization options down the line. For
#       example for both types, networks need a different final activation function...
DATASET_TYPE: str = 'classification'  # 'regression'

# == STUDENT TEACHER PARAMETERS ==
# These are the parameters which are directly relevant to the process of performing the student teacher
# analysis.
REPETITIONS: int = 10
STUDENT_KWARGS = {
    'units': [5, 5, 5],
    'concat_heads': False,
    'importance_channels': 2,
    'final_units': [2],
    'sparsity_factor': 1,
    'use_graph_attributes': False,
    'final_activation': 'softmax'
}
HYPER_KWARGS = {
    'epochs': 100,
    'batch_size': 8,
    'optimizer_cb': lambda: ks.optimizers.Nadam(learning_rate=0.01),
    'prediction_metric_cb': lambda: ks.metrics.CategoricalAccuracy(),
    'importance_metric_cb': lambda: ks.metrics.MeanAbsoluteError(),
    'log_progress': 10,
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
                            ExplanationLoss(),
                            ExplanationLoss()],
        'loss_weights': [1, 0.1, 0.1]
    },
}
VARIANTS = list(VARIANT_KWARGS.keys())
DEVICE: str = 'cpu:0'

# == EVALUATION PARAMETERS ==
# :param PREDICTION_METRIC_KEY:
#       The string name of the metric which is calculated during the training process.
PREDICTION_METRIC_KEY: str = 'categorical_accuracy'
LOG_STEP_EVAL: int = 100
# :param VARIANT_COLORS:
#       A dict whose keys are the string names of the different variants and the values should be valid
#       matplotlib color identifiers, which will be used to represent the corresponding variants in all
#       the plots that will be created.
VARIANT_COLORS: dict = {
    'ref': 'blue',
    'exp': 'green',
}
# :param PLOT_TRAINING:
#       boolean flag for whether to create plots about the model training processes (loss and metric over
#       training epochs)
PLOT_TRAINING: bool = True

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/vgd_student'
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    # :hook start_experiment:
    #       This hook can be used to inject additional code at the very start of the experiment for example
    #       to modify some parameters based on some dynamic condition.
    e.apply_hook('start_experiment')

    e.info('starting to perform student teacher analysis...')

    e.info('loading visual graph dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )
    dataset_size = len(index_data_map)

    e.info('loading visual graph dataset expansions...')
    for expansion_path in VISUAL_GRAPH_DATASET_EXPANSION_PATHS:
        expansion_name = os.path.basename(expansion_path)
        e.info(f'loading expansion "{expansion_name}"...')
        load_visual_graph_dataset_expansion(
            index_data_map=index_data_map,
            expansion_path=expansion_path,
            logger=e.logger,
            log_step=LOG_STEP_EVAL
        )

    e.info('extracting graph representations from visual graph dataset...')
    dataset: t.List[dict] = [None for _ in range(dataset_size)]
    dataset_indices: t.List[int] = []
    for index, data in index_data_map.items():
        metadata = data['metadata']

        g = data['metadata']['graph']
        if HAS_EXPLANATIONS:
            g['node_importances'] = g[f'node_importances_{EXPLANATION_POSTFIX}']
            g['edge_importances'] = g[f'edge_importances_{EXPLANATION_POSTFIX}']
        else:
            g['node_importances'] = np.zeros(shape=(len(g['node_indices']), EXPLANATION_CHANNELS))
            g['edge_importances'] = np.zeros(shape=(len(g['edge_indices']), EXPLANATION_CHANNELS))

        dataset[index] = g
        # Now we apply the filter mechanism on the INDICES instead of the dataset itself. We will filter
        # the "dataset_indices" list so that the train and test indices will be derived from that filtered
        # version. This will be enough to effectively restrict the dataset.
        if DATASET_FILTER_KEY is not None:
            if (DATASET_FILTER_KEY not in metadata) or (not metadata[DATASET_FILTER_KEY]):
                continue

        dataset_indices.append(index)

    # We need those later
    e['dataset_indices'] = dataset_indices
    dataset_size = len(dataset)
    dataset_indices_set = set(dataset_indices)
    e.info(f'loaded dataset with {dataset_size} elements')

    @e.hook('sweep_generator', default=True)
    def sweep_generator_function(_e, dataset):
        yield 'main'


    e.info(f'starting with the analysis loop...')
    e['sweep_keys'] = []
    e['REPETITIONS'] = REPETITIONS
    for key in e.apply_hook('sweep_generator', dataset=dataset):
        e['sweep_keys'].append(key)

        @e.hook('create_student_template', default=True)
        def create_student_template(_e):
            return StudentTemplate(
                student_class=MeganStudent,
                student_name=f'megan',
                **_e.p['STUDENT_KWARGS']
            )

        e.info(f'setting up the student template and analysis...')
        # :hook create_student_template:
        #       This function should return a StudentTemplate object instance for the keras based student to
        #       be used for the analysis.
        student_template = e.apply_hook('create_student_template')

        student_teacher_analysis = StudentTeacherExplanationAnalysis(
            student_template=student_template,
            trainer_class=KerasStudentTrainer,
            results_class=AbstractStudentResult,
            same_weights=True,
            logger=e.logger
        )

        for rep in range(REPETITIONS):
            e.info(f'REPETITION {rep+1}/{REPETITIONS}')

            e.info(f'creating train test split...')
            if 'overwrite_indices' in e.data:
                e.info('detected overwrite indices, using those instead...')
                dataset_indices = e['overwrite_indices']

            dataset_indices_set = set(dataset_indices)
            k_train = int(TRAIN_RATIO * len(dataset_indices))
            if TRAIN_NUM is not None:
                k_train = TRAIN_NUM
            train_indices = random.sample(dataset_indices, k=k_train)
            train_indices_set = set(train_indices)
            test_indices_set = dataset_indices_set.difference(train_indices_set)
            #test_indices = list(test_indices_set)
            test_indices = [i for i in dataset_indices if i not in train_indices_set]

            # 01.04.2023 - For the GNES student, the issue arose that it really does not handle large
            # test sets well and usually the test set size if overkill anyway, so this optional feature
            # will enable only using a random subset of the test set for the process.
            if TEST_SUBSET is not None:
                num_subset = int(TEST_SUBSET * len(test_indices))
                test_indices = random.sample(test_indices, k=num_subset)

            e[f'train_indices/{key}/{rep}'] = train_indices
            e[f'test_indices/{key}/{rep}'] = test_indices
            e.info(f'selected {len(train_indices)} train, {len(test_indices)} test')

            e.info(f'starting the analysis...')
            with tf.device(DEVICE):
                results = student_teacher_analysis.fit(
                    dataset=dataset,
                    hyper_kwargs={
                        **HYPER_KWARGS,
                        'train_indices': train_indices,
                        'test_indices': test_indices,
                    },
                    variant_kwargs=VARIANT_KWARGS,
                    suffix=f'{key}_{rep}'
                )

            for variant, variant_results in results.items():
                # 02.03.2023 - Here we calculate the final prediction performance as the average over the
                # last ten epochs this has been proven to be more consistent as there tend to be serious
                # fluctuations in the test set performance.
                # 01.04.2023 - We can only do that if we actually have a history of validation metrics,
                # which is not always the case! In case we don't have that we just use the default
                # performance value based on the last epoch only
                if 'val' in variant_results['history']:
                    performance = np.mean(variant_results['history']['val']['metric']['out'][-10:])
                    variant_results['metrics']['performance'] = performance

                # 04.03.2023 - Without this, the value will not be properly saved into the JSON experiment
                # storage...
                variant_results['metrics']['performance'] = float(variant_results['metrics']['performance'])

                e.info(f' * {variant}'
                       f' - train_loss: {variant_results["history"]["train"]["loss"]["out"][-1]}'
                       f' - perf: {variant_results["metrics"]["performance"]}'
                       f' - node_auc: {variant_results["metrics"]["node_auc"]}'
                       f' - edge_auc: {variant_results["metrics"]["edge_auc"]}')

            # ~ Visualizing the results

            # 01.04.2023 - Made the training plotting optional, because the GNES student does not support
            # this feature (raises error) and there needed to be a method to not execute this piece of code
            # for certain sub experiments.
            if PLOT_TRAINING:
                e.info('visualizing results...')
                pdf_path = os.path.join(e.path, f'training_{key}_{rep:02d}.pdf')
                with PdfPages(pdf_path) as pdf:
                    # This first plot will show the loss over time for all of the variants.
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                    fig.suptitle('Loss over Epochs')
                    for variant in VARIANTS:
                        ax.plot(
                            results[variant]['epochs'],
                            results[variant]['history']['train']['metric']['out'],
                            color=VARIANT_COLORS[variant],
                            ls='-',
                            label=f'train {variant}'
                        )
                        ax.plot(
                            results[variant]['epochs'],
                            results[variant]['history']['val']['metric']['out'],
                            color=VARIANT_COLORS[variant],
                            ls='--',
                            label=f'test {variant}'
                        )

                    fig.legend()
                    pdf.savefig(fig)
                    plt.close(fig)

            # 29.03.2023 - Here we delete some data from the results dictionary before committing it to
            # the experiment storage. This is because there were some serious memory issues for some
            # larger experiments recently!
            for variant in VARIANTS:
                del results[variant]['model']
                del results[variant]['initial_weights']
                del results[variant]['history']
                del results[variant]['test']

            e[f'results/{key}/{rep}'] = results

            e.status()

        # Printing metrics about the currently finished sweep key
        data_dict = {variant: [e[f'results/{key}/{rep}/{variant}/metrics/performance']
                               for rep in range(REPETITIONS)]
                     for variant in VARIANTS}
        metric_type = 'falling' if DATASET_TYPE == 'regression' else 'rising'
        value = calculate_sts(data_dict['exp'], data_dict['ref'], metric_type=metric_type)
        stats = ttest_rel(data_dict['exp'], data_dict['ref'])

        e[f'sts/{key}'] = value
        e[f'p/{key}'] = stats.pvalue

        e.info(f'finished sweep {key}')
        e.info(f'  * sts: {value:.2f}')
        e.info(f'  * p_value: {stats.pvalue:.2f}')

        exp_median = np.median(data_dict['exp'])
        ref_median = np.median(data_dict['ref'])
        pair_median = np.median(np.array(data_dict['ref']) - np.array(data_dict['exp']))
        e.info(f'  * medians:\n'
               f'    ~ exp: {exp_median:.3f}\n'
               f'    ~ ref: {ref_median:.3f}\n'
               f'    ~ diff: {ref_median - exp_median:.3f}\n'
               f'    ~ pair: {pair_median:.3f}')


with Skippable(), e.analysis:

    REPETITIONS = e['REPETITIONS']

    e.info('printing the final results...')
    for key in e['sweep_keys']:
        e.info('')
        sts = e[f"sts/{key}"]
        p_value = e[f"p/{key}"]
        e.info(f' * {key} - sts: {sts:.3f} - p: {p_value:.3f}')
        for variant in VARIANTS:
            e.info(f'   > {variant} ')

            performances = [e[f'results/{key}/{rep}/{variant}/metrics/performance']
                            for rep in range(REPETITIONS)]
            node_aucs = [e[f'results/{key}/{rep}/{variant}/metrics/node_auc']
                         for rep in range(REPETITIONS)]
            edge_aucs = [e[f'results/{key}/{rep}/{variant}/metrics/edge_auc']
                         for rep in range(REPETITIONS)]

            e.info(f'       ~ performance: {np.mean(performances):.3f} ({np.std(performances):.3f})')
            e.info(f'       ~ node auc: {np.mean(node_aucs):.3f} ({np.std(node_aucs):.3f})')
            e.info(f'       ~ edge auc: {np.mean(edge_aucs):.3f} ({np.std(edge_aucs):.3f})')

    e.info('visualizing student teacher results...')
    pdf_path = os.path.join(e.path, 'results.pdf')
    with PdfPages(pdf_path) as pdf:
        # The first page will be the student teacher plot w.r.t. the main prediction performance
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        fig.suptitle(f'Final Prediction Metric Distributions\n'
                     f'({REPETITIONS} Repetitions)')
        data_list = []
        for key in e['sweep_keys']:
            data_dict = {variant: [e[f'results/{key}/{rep}/{variant}/metrics/performance']
                                   for rep in range(REPETITIONS)]
                         for variant in VARIANTS}
            data_list.append(data_dict)

        centers = plot_multi_variant_box_plot(
            ax=ax,
            data_list=data_list,
            variant_colors=VARIANT_COLORS,
            do_white_background=True,
            plot_between=True,
        )
        y_min, y_max = ax.get_ylim()
        ax.set_ylim([y_min, y_max * 1.1])
        y_text = y_max * 1.05
        for key, center, data_dict in zip(e['sweep_keys'], centers, data_list):
            value = e[f'sts/{key}']
            p_value = e[f'p/{key}']
            if p_value < 0.05:
                string = f'${value:.2f}' + r'^{(*)}$'
            else:
                string = f'${value:.2f}$'

            ax.text(
                center, y_text,
                s=string,
                usetex=True,
                fontsize=15,
                va='center',
                ha='center',
            )

        ax.set_ylabel('Prediction Metric')
        ax.set_xlabel('Sweep Parameter')
        ax.set_xticklabels(e['sweep_keys'])
        pdf.savefig(fig)
        plt.close(fig)

        # The second page will be the same multi box plot but w.r.t. to the explanation accuracy (node auc)
        fig, rows = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), squeeze=False)
        fig.suptitle(f'explanation accuracy distributions\n'
                     f'({REPETITIONS} repetitions)')
        for i, name in enumerate(['node', 'edge']):
            ax = rows[0][i]
            ax.set_title(f'for {name} accuracy')
            data_list = []
            for key in e['sweep_keys']:
                data_list.append(
                    {variant: [e[f'results/{key}/{rep}/{variant}/metrics/{name}_auc']
                               for rep in range(REPETITIONS)]
                     for variant in VARIANTS}
                )

            plot_multi_variant_box_plot(
                ax=ax,
                data_list=data_list,
                variant_colors=VARIANT_COLORS,
                do_white_background=True
            )
            ax.set_ylabel(f'{name} AUC')
            ax.set_xlabel(f'sweep parameter')
            ax.set_xticklabels(e['sweep_keys'])

        pdf.savefig(fig)
        plt.close(fig)

        # This third page will visualize the correlation between the prediction performance and the
        # explanation accuracy.
        fig, rows = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), squeeze=False)
        fig.suptitle('correlation between prediction performance and explanation accuracy')
        for i, name in enumerate(['node', 'edge']):
            ax = rows[0][i]
            ax.set_ylabel(f'{name} AUC')
            ax.set_xlabel('prediction performance')
            # In these lists we will store the entirety of all the values, no matter what variant and what
            # sweep key. These lists will later be used to calculate the overall correlation coefficient
            xs_all: t.List[float] = []
            ys_all: t.List[float] = []
            for key, marker in zip(e['sweep_keys'], ['o', '^', 'x', '*', '+']):
                print(key, marker)
                for variant in VARIANTS:
                    xs = [e[f'results/{key}/{rep}/{variant}/metrics/performance']
                          for rep in range(REPETITIONS)]
                    xs_all += xs
                    ys = [e[f'results/{key}/{rep}/{variant}/metrics/{name}_auc']
                          for rep in range(REPETITIONS)]
                    ys_all += ys
                    ax.scatter(
                        xs, ys,
                        color=VARIANT_COLORS[variant],
                        marker=marker,
                        alpha=0.5,
                    )

            # Here we calculate the overall correlation coefficient between the two value sets (aka the
            # prediction performances and the explanation accuracies for the same run). The exact value
            # we'll put into the title of the plot
            correlation = float(np.corrcoef(xs_all, ys_all)[0][1])
            e[f'performance_accuracy_correlation/{name}'] = correlation
            ax.set_title(f'for {name} accuracy\n'
                         f'correlation: {correlation:.4f}')

            # To visualize this correlation as well we calculate the best linear fit for this scatter
            # dataset here and draw that linear function into the plot as well.
            a, b = np.polyfit(xs_all, ys_all, deg=1)
            x_min, x_max = ax.get_xlim()
            ax.plot(
                [x_min, x_max],
                [a * x_min + b, a * x_max + b],
                color='black',
                alpha=0.5,
                zorder=1
            )

        pdf.savefig(fig)
        plt.close(fig)

    # 28.03.2023 - Now we also want to create a latex table with the results because we probably won't
    # always present the results as the box plot, but also sometimes just have to present the numerical
    # values.
    e.info('creating a latex table...')
    column_names = [
        'sweep key',
        'STS',
        'p',
    ]
    rows = []
    for key in e['sweep_keys']:
        row = []
        row.append(key.replace('_', ' '))
        row.append(e[f'sts/{key}'])
        row.append(e[f'p/{key}'])

        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_mean,
        caption=f'Results of {REPETITIONS} repetition(s) of Student Teacher Analysis'
    )
    e.commit_raw('table.tex', table)
    pdf_path = os.path.join(e.path, 'table.pdf')
    render_latex(
        {'context': table},
        output_path=pdf_path,
    )

    e.info('saving the experiment data...')
    e.save_experiment_data()
