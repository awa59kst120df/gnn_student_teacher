import copy
import logging
import random
import typing as t
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

from gnn_student_teacher.util import NULL_LOGGER
from gnn_student_teacher.students import StudentTemplate
from gnn_student_teacher.students import AbstractStudentModel
from gnn_student_teacher.students import AbstractStudentTrainer
from gnn_student_teacher.data.structures import AbstractStudentResult


def relu(x):
    return x * (x > 0)


def calculate_sts(exp_results: t.List[float],
                  ref_results: t.List[float],
                  metric_type: str = 'rising'
                  ) -> float:
    if metric_type == 'rising':
        factor = -1
    elif metric_type == 'falling':
        factor = +1
    else:
        raise ValueError(f'The parameter "metric_type" only has two allowed values "rising" and "falling" '
                         f'but was given the value "{metric_type}"!')

    value = float(np.median(factor * (np.array(ref_results) - np.array(exp_results))))

    return value


def cha_srihari_distance(values1: t.List[float],
                         values2: t.List[float],
                         bins: int = 100,
                         value_range: t.Tuple[float, float] = (0, 1),
                         ) -> float:
    """
    https://www.sciencedirect.com/science/article/pii/S0031320301001182
    """
    num_elements = len(values1)
    h1, _ = np.histogram(values1, bins=bins, range=value_range)
    h2, _ = np.histogram(values2, bins=bins, range=value_range)

    prefix_sum = 0
    h_dist = 0
    for i in range(bins):
        prefix_sum += (h1[i] - h2[i]) / bins
        h_dist += abs(prefix_sum)

    return h_dist / num_elements


class StudentTeacherExplanationAnalysis:

    def __init__(self,
                 student_template: StudentTemplate,
                 trainer_class: type,
                 results_class: type,
                 same_weights: bool = False,
                 logger: logging.Logger = NULL_LOGGER):
        self.template = student_template
        self.trainer_class = trainer_class
        self.results_class = results_class
        self.same_weights = same_weights
        self.logger = logger

    def fit(self,
            dataset: t.Any,
            hyper_kwargs: t.Dict[str, t.Any],
            variant_kwargs: t.Dict[str, t.Any],
            suffix: t.Optional[str] = None):

        variants = list(variant_kwargs.keys())
        self.logger.info(f'starting student teacher analysis with variants {variants} ...')
        results = {}
        weights = None
        for variant, kwargs in variant_kwargs.items():
            self.logger.info(f'instantiating new variant "{variant}"...')
            model: AbstractStudentModel = self.template.instantiate(
                variant=variant if suffix is None else f'{variant}_{suffix}',
                additional_kwargs=kwargs['model_kwargs'] if 'model_kwargs' in kwargs else {}
            )

            trainer: AbstractStudentTrainer = self.trainer_class(
                model=model,
                dataset=dataset,
                results_class=self.results_class,
                logger=self.logger,
                initial_weights=weights,
            )

            self.logger.info(f'training variant "{variant}"...')
            trainer_kwargs = self._merge_kwargs([hyper_kwargs, kwargs])
            result: AbstractStudentResult = trainer.fit(**trainer_kwargs)
            results[variant] = dict(result)

            if self.same_weights and weights is None:
                self.logger.info('attempting to acquire initial weights to share among variants...')
                if 'initial_weights' in result.keys():
                    weights = result['initial_weights']
                    self.logger.info(f'The initial weights of the first variant {variant} will be used '
                                     f're-used for all other variants for improved comparability')
                else:
                    self.logger.warning(f'The result dictionary of the training process for variant '
                                        f'{variant} does not have a "initial_weights" field and thus the '
                                        f'same weights cannot be reused!')

        return results

    def fit_multiple(self):
        pass

    @classmethod
    def _merge_kwargs(cls, kwargs_list: t.List[dict]):
        kwargs_merged = {}
        for kwargs in kwargs_list:
            for key, value in kwargs.items():
                if key.endswith('_cb'):
                    value = value()
                    key = key[:-3]

                kwargs_merged[key] = value

        return kwargs_merged
