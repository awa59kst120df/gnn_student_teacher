import typing as t
import logging
import random
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.python.keras.engine import compile_utils
from sklearn.metrics import roc_auc_score

from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.modules import ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.modules import LazyConcatenate, LazyAverage
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.conv.gat_conv import MultiHeadGATV2Layer
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.pooling import PoolingLocalEdges
from graph_attention_student.models import Megan
from graph_attention_student.models import GnesGradientModel
from graph_attention_student.models import grad_importances

from gnn_student_teacher.util import NULL_LOGGER
from gnn_student_teacher.students import AbstractStudentModel
from gnn_student_teacher.students import AbstractStudentTrainer
from gnn_student_teacher.layers import ExplanationSparsityRegularization
from gnn_student_teacher.data.structures import AbstractStudentResult
from gnn_student_teacher.training.callbacks import LogProgressCallback
from gnn_student_teacher.training.losses import shifted_sigmoid


class KerasStudentModel(AbstractStudentModel, ks.models.Model):

    def __init__(self,
                 name: str,
                 variant: str,
                 **kwargs):
        AbstractStudentModel.__init__(self, name, variant)
        ks.models.Model.__init__(self, name=self.full_name, **kwargs)

    def predict_single(self,
                       node_attributes: np.ndarray,
                       edge_attributes: np.ndarray,
                       edge_indices: np.ndarray
                       ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out, ni, ei = self([
            ragged_tensor_from_nested_numpy([node_attributes]),
            ragged_tensor_from_nested_numpy([edge_attributes]),
            ragged_tensor_from_nested_numpy([edge_indices])
        ])

        return (
            out.numpy()[0],
            ni.numpy()[0],
            ei.numpy()[0]
        )


class MockKerasStudent(KerasStudentModel):

    def __init__(self,
                 name: str,
                 variant: str,
                 hidden_units: int,
                 output_units: int,
                 importance_units: int,
                 **kwargs):
        KerasStudentModel.__init__(self, name, variant, **kwargs)
        self.lay_conv = GCN(units=hidden_units)
        self.lay_pooling = PoolingNodes(pooling_method='sum')
        self.lay_dense = DenseEmbedding(units=output_units)
        self.lay_dense_nodes = DenseEmbedding(units=importance_units)
        self.lay_dense_edges = DenseEmbedding(units=importance_units)

    def call(self, inputs, training=False):
        node_input, edge_input, edge_index = inputs

        x = self.lay_conv([node_input, edge_input, edge_index])
        x = self.lay_pooling(x)
        x = self.lay_dense(x)
        ni = ks.backend.sigmoid(self.lay_dense_nodes(node_input))
        ei = ks.backend.sigmoid(self.lay_dense_edges(edge_input))

        return x, ni, ei


class KerasStudentTrainer(AbstractStudentTrainer):
    """
    A concrete implementation of an abstract student trainer, specifically for GNN student models implemented
    with keras (KGCNN - https://github.com/aimat-lab/gcnn_keras).

    A trainer instance uses a given kgcnn ``model`` instance and a ``dataset`` for a training process which
    is executed with :func:`~fit`.
    """

    DATASET_REQUIRED_FIELDS = {
        'node_attributes': {
            'type': (list, np.ndarray),
            'dtype': np.dtype(float)
        },
        'edge_attributes': {
            'type': (list, np.ndarray),
            'dtype': np.dtype(float)
        },
        'edge_indices': {
            'type': (list, np.ndarray),
            'dtype': np.dtype(int),
        },
        'graph_labels': {
            'type': (list, np.ndarray),
            'dtype': np.dtype(float),
        },
        'node_importances': {
            'type': (list, np.ndarray),
            'dtype': np.dtype(float)
        },
        'edge_importances': {
            'type': (list, np.ndarray),
            'dtype': np.dtype(float)
        }
    }

    def __init__(self,
                 model: KerasStudentModel,
                 dataset: t.Union[dict, list],
                 logger: logging.Logger = NULL_LOGGER,
                 dataset_required_fields: dict = {},
                 initial_weights: t.Optional[list] = None,
                 **kwargs):
        AbstractStudentTrainer.__init__(self, model, dataset, logger, **kwargs)
        self.initial_weights = initial_weights

        # self.dataset_required_fields is a dictionary which defines the fields which each GraphDict of
        # the dataset absolutely has to have.
        # This dict is used to validate that the "dataset" list has the correct format for the training
        # process.
        self.dataset_required_fields = self.DATASET_REQUIRED_FIELDS.copy()
        self.dataset_required_fields.update(dataset_required_fields)

        # ~ Additional attributes that will be set in "compile"
        self.dataset_validated: t.Optional[t.Dict[str, t.List[np.ndarray]]] = None
        self.x_train: t.Optional[t.Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]] = None
        self.x_test: t.Optional[t.Tuple[np.ndarray, tf.RaggedTensor, tf.RaggedTensor]] = None
        self.y_train: t.Optional[t.Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]] = None
        self.y_test: t.Optional[t.Tuple[np.ndarray, tf.RaggedTensor, tf.RaggedTensor]] = None

        self.test_indices = None
        self.result = None

    def __fit__(self,
                result: AbstractStudentResult,
                optimizer: ks.optimizers.Optimizer,
                loss: t.List[ks.losses.Loss],
                loss_weights: t.List[float],
                prediction_metric: ks.metrics.Metric,
                importance_metric: ks.metrics.Metric,
                batch_size: int = 32,
                epochs: int = 100,
                test_split: float = 0.1,
                test_indices: t.Optional[t.List[int]] = None,
                train_indices: t.Optional[t.List[int]] = None,
                log_progress: t.Optional[int] = 10,
                callbacks: t.List[ks.callbacks.Callback] = [],
                run_eagerly: bool = False,
                do_eval: bool = True,
                test_batch_size: t.Optional[bool] = None,
                **kwargs):
        """
        This method actually executes the training process.

        01.04.2023 - (1) Added the boolean flag "do_eval", which determines if the keras fit() evaluation
            step should be used at all during the model training. This is necessary to support the GNES
            student, which does not support the eval step during training! (2) Also added the optional
            argument "test_batch_size" which if given will cause the final evaluation on the test set to
            be done in a batched manner. (3) Lots of internal refactoring.

        :param result: An AbstractStudentResult instance. This is essentially used as a dictionary, where
            all the results of the training process will be stored into. This instance which is given as the
            parameter will be modified by this method and then also returned in the end.
        :param optimizer: A keras Optimizer instance which should be used to calculate and apply the
            gradients for the training process.
        :param loss: A list of 3 keras Loss instances. The first loss is the main prediction loss and the
            second and third loss are for the node and edge importance explanation tensors respectively.
            Note that these usually have to be different from the prediction loss!
        :param loss_weights: A list of 3 float values, each one representing the weighting factor of the
            corresponding loss instance.
        :param batch_size: The integer number of elements to be used for one batch of the training process
        :param epochs: The number of epochs for which to the model is trained.
        :param test_indices: An array of integer dataset indices defining those elements to be used as the
            test set of the training process.
        :param train_indices: An array of integer dataset indices defining those elements to be used as the
            train set of the training process.
        :param test_split: If ``test_indices`` is NOT given, then this argument can be used to define the
            relative size of the test set instead. The train-test split of the indices will then be
            randomly generaged accordingly.
        :param do_eval: A boolean flag of whether the evaluation step during the keras fit() function should
            be executed at all. Default is True. Setting this flag to False can be useful if a student model
            does not support the evaluation step or to achieve superior efficiency. Note that a test set
            evaluation for the final epoch will still be performed in any way!
        :param test_batch_size: Optional integer batch size for performing the prediction on the final test
            set. If this is not set, the model will be queried with all the test data at once. If an integer
            is given, that prediction will be made in batches instead.

        :returns: The AbstractStudentResult instance which was given as an argument and which was extended
            with the results of the training process over the course of this method.
        """
        self.result = result

        # ~ Preparing the dataset

        # Coming into this function, the dataset is allowed to have different formats. For example it could
        # be a dict with list values or it could be a list of many dictionaries. This method makes sure that
        # all the required fields exist for all the elements of the dataset and then turns all the different
        # formats into one universal format, which is a dict that maps from the string field names to lists
        # of numpy arrays, where each array represents one element of the dataset. This is the format
        # required to further process the dataset into the ragged tensors for model training.
        self.dataset_validated: t.Dict[str, list] = self._validate_dataset(self.dataset)

        self.indices = list(range(len(self.dataset)))
        self.x, self.y = self.dataset_to_tensors(self.dataset_validated, self.indices)

        # Here we turn the dataset into the actual tuples of keras RaggedTensor objects which are required
        # for the model training process
        if test_indices is not None:
            dataset_split = self._split_dataset_indices(self.dataset_validated, test_indices, train_indices)
        else:
            test_indices = self._random_test_indices(self.dataset_validated, test_split)
            dataset_split = self._split_dataset_indices(self.dataset_validated, test_indices, train_indices)

        self.test_indices = test_indices
        num_test = len(test_indices)
        self.x_train, self.y_train, self.x_test, self.y_test = dataset_split

        # ~ Preparing the model
        self.logger.info(f'compiling model "{self.model.full_name}"...')

        # 01.04.2023 - The
        if do_eval:
            metrics = [prediction_metric, importance_metric]
        else:
            metrics = []

        self.model.compile(
            loss=loss,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            run_eagerly=run_eagerly
        )

        # 25.02.2023
        # By piping sample input through the model we actually build it, such that it's weights get
        # initialized. We do that here, so we can put its initial weights into the result dict. These initial
        # weights may be needed by the overall student teacher process.
        self.model(self.x_test)
        if self.initial_weights is not None:
            self.model.set_weights(self.initial_weights)

        result['initial_weights'] = self.model.get_weights()

        # Preparing the callbacks
        if log_progress is not None:
            log_progress_callback = LogProgressCallback(
                logger=self.logger,
                epoch_step=log_progress,
                key=[
                    f'val_output_1_{prediction_metric.name}',
                    'loss',
                    'output_1_loss',
                    'output_2_loss',
                    'output_3_loss',
                ]
            )
            callbacks.append(log_progress_callback)

        start_time = time.time()
        self.logger.info(f'starting training of model "{self.model.full_name}"...')
        self.logger.info(
            f'training config'
            f' - epochs: {epochs}'
            f' - batch_size: {batch_size}'
        )
        # 01.04.2023 - The
        if do_eval:
            validation_freq = 1
            validation_data = (self.x_test, self.y_test)
        else:
            validation_freq = 1
            validation_data = None

        hist = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_freq=validation_freq,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'finished training of model "{self.model.full_name}" '
                         f'in {duration:.1f}s ({duration / 3600:.1f}h)')

        result['duration'] = duration
        result['epochs'] = list(range(epochs))
        result['model'] = self.model
        result['test_indices'] = test_indices

        # Training history
        result['history/train/loss/out'] = hist.history[f'output_1_loss']
        result['history/train/loss/ni'] = hist.history[f'output_2_loss']
        result['history/train/loss/ei'] = hist.history[f'output_3_loss']
        if do_eval:
            result['history/train/metric/out'] = hist.history[f'output_1_{prediction_metric.name}']
            result['history/train/metric/ni'] = hist.history[f'output_2_{importance_metric.name}']
            result['history/train/metric/ei'] = hist.history[f'output_3_{importance_metric.name}']

        # Validation history
        if do_eval:
            result['history/val/loss/out'] = hist.history[f'val_output_1_loss']
            result['history/val/loss/ni'] = hist.history[f'val_output_2_loss']
            result['history/val/loss/ei'] = hist.history[f'val_output_3_loss']
            result['history/val/metric/out'] = hist.history[f'val_output_1_{prediction_metric.name}']
            result['history/val/metric/ni'] = hist.history[f'val_output_2_{importance_metric.name}']
            result['history/val/metric/ei'] = hist.history[f'val_output_3_{importance_metric.name}']

        # ~ Evaluating on the test set
        # After the training is completed, we want to evaluate the final model on the unseen test data. For
        # this process we need to differentiate if it should be in a batched manner or if all the test data
        # should be queried into the model all at once (there may be models which have a severe problem with
        # that!)
        # In either version, these following methods will predict the target values and the explanations for
        # all the elements in the test set and then save the results into the "test" key of the results
        # instance...
        if test_batch_size is not None:
            self.logger.info(f'batched evaluation of final epoch on test set, '
                             f'with batch size {test_batch_size}...')
            self.model_predict_batched(self.test_indices, test_batch_size)
        else:
            self.logger.info('evaluation of final epoch on test set...')
            self.model_predict(self.test_indices)

        # ~ Calculating additional metrics
        # Based on the predictions on the test set, we want to calculate some metrics. The most important
        # ones are these:
        # (1) performance: The target prediction performance. The concrete metric here may differ for
        #     classification and regression tasks.
        # (2) explanation accuracy: How well do the explanations generated by the student model match the
        #     "teacher" explanations from the dataset

        result['metrics/performance'] = prediction_metric(
            [result[f'test/out/true/{index}'] for index in test_indices],
            [result[f'test/out/pred/{index}'] for index in test_indices]
        )

        ni_true = self.y_test[1]
        ni_true_flat = [round(v)
                        for a in ni_true.numpy()
                        for v in a.flatten()]
        ni_pred_flat = [v
                        for index in self.test_indices
                        for v in np.array(result[f'test/ni/{index}']).flatten()]
        result['metrics/node_auc'] = roc_auc_score(
            ni_true_flat,
            ni_pred_flat,
        )

        ei_true = self.y_test[2]
        ei_true_flat = [round(v)
                        for a in ei_true.numpy()
                        for v in a.flatten()]
        ei_pred_flat = [v
                        for index in self.test_indices
                        for v in np.array(result[f'test/ei/{index}']).flatten()]
        result['metrics/edge_auc'] = roc_auc_score(
            ei_true_flat,
            ei_pred_flat,
        )

        # An attempt to reset the entire keras graph so that the next "fit" process is not polluted with
        # side effects of the current run.
        ks.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        if log_progress is not None:
            log_progress_callback.active = False

        return result

    def model_predict(self,
                      indices: t.List[int],
                      call_kwargs: dict = {}
                      ) -> None:
        """
        This method will query the ``self.model`` using a subset of the dataset defined by the
        ``indices`` list of integer dataset indices. The resulting predictions as well as the resulting
        explanations are saved into the ``self.results`` results instance.

        :param indices: A list of indices to be used as the indices,
        :param call_kwargs: Optional a dictionary which can be used to pass additional keyword
            arguments to the invocation of the models __call__ method.

        :returns: None
        """
        x_gathered = [tf.gather(v, indices=indices) for v in self.x]
        out, ni, ei = self.model(x_gathered, **call_kwargs)
        for c, index in enumerate(indices):
            y_true = self.dataset_validated['graph_labels'][index].tolist()
            self.result[f'test/out/true/{index}'] = y_true

            y_pred = out[c].numpy().tolist()
            self.result[f'test/out/pred/{index}'] = y_pred
            self.result[f'test/ni/{index}'] = ni[c].numpy().tolist()
            self.result[f'test/ei/{index}'] = ei[c].numpy().tolist()

        return self.result

    def model_predict_batched(self,
                              indices: t.List[int],
                              batch_size: int
                              ) -> None:
        """
        This method will query the ``self.model`` using a subset of the dataset defined by the
        ``indices`` list of integer dataset indices. The resulting predictions as well as the resulting
        explanations are saved into the ``self.results`` results instance. This method will do the
        prediction sequentially with batches of the size ``batch_size``.

        :param indices: A list of indices to be used as the indices,
        :param batch_size: The number of elements with which the model is queried during one iteration.

        :returns: None
        """
        max_index = len(indices)
        current_index = 0
        while current_index < max_index:
            num_batch = min(max_index - current_index, batch_size)
            indices_batch = indices[current_index:current_index+num_batch]
            self.model_predict(
                indices=indices_batch,
                call_kwargs={'batch_size': num_batch}
            )

            current_index += num_batch
            self.logger.info(f' * ({current_index}/{max_index}) done')

    def dataset_to_tensors(self,
                           dataset: t.Dict[str, list],
                           indices: t.List[int]
                           ) -> t.Tuple[tuple, tuple]:
        """
        Given a validated dataset dictionary ``dataset``, this method will compute and return
        the model input tensor x and the output tensor y corresponding to the elements defined by their
        integer indices in the list ``indices``.

        :param dataset: A dictionary whose keys are the string attribute names defined by
            ``self.dataset_required_fields`` and the values are lists of numpy arrays defining the
            corresponding attributes of each of the dataset elements.
        :param indices: A list of integers, which needs to be a subset of all valid dataset indices. This
            list defines which elements of the given ``dataset`` and in which order they will be used to
            create the tensors.

        :returns: A tuple of two elements (x, y) where both elements are tuples themselves, each consisting
            of three tensors which represent the model input and output respectively.
        """
        attribute_map = {}
        for field, info_dict in self.dataset_required_fields.items():
            attribute_map[field] = [dataset[field][i] for i in indices]

        x = (
            ragged_tensor_from_nested_numpy(attribute_map['node_attributes']),
            ragged_tensor_from_nested_numpy(attribute_map['edge_attributes']),
            ragged_tensor_from_nested_numpy(attribute_map['edge_indices'])
        )

        y = (
            ragged_tensor_from_nested_numpy(attribute_map['graph_labels']),
            ragged_tensor_from_nested_numpy(attribute_map['node_importances']),
            ragged_tensor_from_nested_numpy(attribute_map['edge_importances']),
        )

        return x, y

    # -- private methods --

    def _split_dataset_indices(self,
                               dataset: t.Dict[str, t.List[np.ndarray]],
                               test_indices: t.List[int],
                               train_indices: t.Optional[t.List[int]],
                               ) -> t.Tuple[tuple, tuple, tuple, tuple]:
        indices = list(range(len(dataset['graph_labels'])))
        if train_indices is None:
            train_indices = [i for i in indices if i not in test_indices]

        train_map = {}
        test_map = {}
        for field, info_dict in self.dataset_required_fields.items():
            train_map[field] = [dataset[field][i] for i in train_indices]
            test_map[field] = [dataset[field][i] for i in test_indices]

        x_train = (
            ragged_tensor_from_nested_numpy(train_map['node_attributes']),
            ragged_tensor_from_nested_numpy(train_map['edge_attributes']),
            ragged_tensor_from_nested_numpy(train_map['edge_indices'])
        )

        x_test = (
            ragged_tensor_from_nested_numpy(test_map['node_attributes']),
            ragged_tensor_from_nested_numpy(test_map['edge_attributes']),
            ragged_tensor_from_nested_numpy(test_map['edge_indices'])
        )

        y_train = (
            np.array(train_map['graph_labels']),
            ragged_tensor_from_nested_numpy(train_map['node_importances']),
            ragged_tensor_from_nested_numpy(train_map['edge_importances'])
        )

        y_test = (
            np.array(test_map['graph_labels']),
            ragged_tensor_from_nested_numpy(test_map['node_importances']),
            ragged_tensor_from_nested_numpy(test_map['edge_importances'])
        )

        return x_train, y_train, x_test, y_test

    def _random_test_indices(self,
                             dataset: t.Dict[str, t.List[np.ndarray]],
                             test_split: float,
                             ):
        indices = list(range(len(dataset['graph_labels'])))
        dataset_length = len(indices)
        test_count = int(dataset_length * test_split)
        test_indices = random.sample(indices, k=test_count)
        return test_indices

    def _validate_dataset(self,
                          dataset: t.Union[dict, list]
                          ) -> t.Dict[str, list]:
        if isinstance(dataset, dict):
            return self._validate_dataset_dict(dataset)
        elif isinstance(dataset, list):
            return self._validate_dataset_list(dataset)

    def _validate_dataset_list(self,
                               dataset: t.List[dict]
                               ) -> t.Dict[str, list]:
        dataset_validated = defaultdict(list)

        for index, data in enumerate(dataset):
            for field, info_dict in self.dataset_required_fields.items():
                assert field in data, (
                    f'Element {index} of the dataset does not contain the field {field}!'
                )

                assert isinstance(data[field], info_dict['type']), (
                    f'Field {field} of element {index} of the dataset is none of the correct types: '
                    f'{info_dict["type"]}'
                )

                # We need to make sure that the elements of the dataset dict are numpy arrays because that
                # is the format which we will need to turn them into ragged tensors later on...
                dataset_validated[field].append(
                    np.array(data[field], dtype=info_dict['dtype'])
                )

        return dict(dataset_validated)

    def _validate_dataset_dict(self,
                               dataset: t.Dict[str, t.Union[list, np.ndarray]]
                               ) -> t.Dict[str, list]:
        dataset_validated = {}

        for field, info_dict in self.dataset_required_fields.items():
            assert field in dataset.keys(), (
                f'Dataset does not contain the required field {field}!'
            )
            assert isinstance(dataset[field], info_dict['type']), (
                f'Dataset field "{field}" needs to be of type {info_dict["type"]}'
            )

            # We need to make sure to convert all the potential lists that make up the dataset into numpy
            # arrays because this is the format that we later need to convert the whole dataset into the
            # tensorflow ragged tensors which we need for the training process of the models.
            dataset_validated[field] = [np.array(element, dtype=info_dict['dtype'])
                                        for element in dataset[field]]

        return dataset_validated


# == ACTUAL STUDENT IMPLEMENTATIONS ==

class GnesStudent(KerasStudentModel, GnesGradientModel):
    """

    """
    def __init__(self,
                 name: str,
                 variant: str,
                 units: t.List[int],
                 batch_size: int,
                 final_units: t.List[int],
                 final_activation: str,
                 importance_func: t.Callable = grad_importances,
                 layer_cb: t.Callable = lambda k: AttentionHeadGATV2(
                    k,
                    use_edge_features=True,
                    use_final_activation=True,
                    use_bias=True
                    ),
                 ):
        KerasStudentModel.__init__(
            self,
            name=name,
            variant=variant,
        )

        GnesGradientModel.__init__(
            self,
            units=units,
            batch_size=batch_size,
            final_units=final_units,
            final_activation=final_activation,
            importance_func=importance_func,
            layer_cb=layer_cb,
        )


class MeganStudent(KerasStudentModel, Megan):

    def __init__(self,
                 # student parameters
                 name: str,
                 variant: str,
                 # model parameters
                 units: t.List[int],
                 dropout_rate: float = 0.0,
                 use_bias: bool = True,
                 use_edge_features: bool = True,
                 concat_heads: bool = False,
                 importance_channels: int = 1,
                 importance_factor: float = 0.0,
                 importance_multiplier: float = 5.0,
                 importance_transformations: t.Optional[t.List[ks.layers.Layer]] = None,
                 sparsity_factor: float = 0.0,
                 final_units: t.List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 regression_reference: t.Optional = None,
                 regression_limits: t.Optional = None,
                 use_graph_attributes: bool = False,
                 **kwargs):
        KerasStudentModel.__init__(
            self,
            name=name,
            variant=variant,
            **kwargs
        )
        Megan.__init__(
            self,
            units=units,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_edge_features=use_edge_features,
            concat_heads=concat_heads,
            importance_channels=importance_channels,
            importance_factor=importance_factor,
            importance_multiplier=importance_multiplier,
            importance_transformations=importance_transformations,
            sparsity_factor=sparsity_factor,
            final_units=final_units,
            final_dropout_rate=final_dropout_rate,
            final_activation=final_activation,
            regression_reference=regression_reference,
            regression_limits=regression_limits,
            use_graph_attributes=use_graph_attributes,
            **kwargs,
        )

# == DEPRECATED ==

class _MeganStudent(KerasStudentModel):
    """
    MEGAN: Multi Explanation Graph Attention Network
    This model currently supports graph regression and graph classification problems. It was mainly designed
    with a focus on explainable AI (XAI). Along the main prediction, this model is able to output multiple
    attention-based explanations for that prediction. More specifically, the model outputs node and edge
    attributional explanations (assigning [0, 1] values to ever node / edge of the input graph) in K
    separate explanation channels, where K can be chosen as an independent model parameter.
    """

    def __init__(self,
                 # student implementation
                 name: str,
                 variant: str,
                 # convolutional network related arguments
                 units: t.List[int],
                 activation: str = "kgcnn>leaky_relu",
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 use_edge_features: bool = True,
                 # node/edge importance related arguments
                 importance_units: t.List[int] = [],
                 importance_channels: int = 2,
                 importance_activation: str = "sigmoid",  # do not change
                 importance_dropout_rate: float = 0.0,  # do not change
                 importance_factor: float = 0.0,
                 importance_multiplier: float = 10.0,
                 sparsity_factor: float = 0.0,
                 concat_heads: bool = True,
                 # mlp tail end related arguments
                 final_units: t.List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 final_pooling: str = 'sum',
                 regression_limits: t.Optional[t.Tuple[float, float]] = None,
                 regression_reference: t.Optional[float] = None,
                 return_importances: bool = True,
                 **kwargs):
        """
        Args:
            units: A list of ints where each element configures an additional attention layer. The numeric
                value determines the number of hidden units to be used in the attention heads of that layer
            activation: The activation function to be used within the attention layers of the network
            use_bias: Whether the layers of the network should use bias weights at all
            dropout_rate: The dropout rate to be applied after *each* of the attention layers of the network.
            use_edge_features: Whether edge features should be used. Generally the network supports the
                usage of edge features, but if the input data does not contain edge features, this should be
                set to False.
            importance_units: A list of ints where each element configures another dense layer in the
                subnetwork that produces the node importance tensor from the main node embeddings. The
                numeric value determines the number of hidden units in that layer.
            importance_channels: The int number of explanation channels to be produced by the network. This
                is the value referred to as "K". Note that this will also determine the number of attention
                heads used within the attention subnetwork.
            importance_factor: The weight of the explanation-only train step. If this is set to exactly
                zero then the explanation train step will not be executed at all (less computationally
                expensive)
            importance_multiplier: An additional hyperparameter of the explanation-only train step. This
                is essentially the scaling factor that is applied to the values of the dataset such that
                the target values can reasonably be approximated by a sum of [0, 1] importance values.
            sparsity_factor: The coefficient for the sparsity regularization of the node importance
                tensor.
            concat_heads: Whether to concat the heads of the attention subnetwork. The default is True. In
                that case the output of each individual attention head is concatenated and the concatenated
                vector is then used as the input of the next attention layer's heads. If this is False, the
                vectors are average pooled instead.
            final_units: A list of ints where each element configures another dense layer in the MLP
                at the tail end of the network. The numeric value determines the number of the hidden units
                in that layer. Note that the final element in this list has to be the same as the dimension
                to be expected for the samples of the training dataset!
            final_dropout_rate: The dropout rate to be applied after *every* layer of the final MLP.
            final_activation: The activation to be applied at the very last layer of the MLP to produce the
                actual output of the network.
            final_pooling: The pooling method to be used during the global pooling phase in the network.
            regression_limits: A tuple where the first value is the lower limit for the expected value range
                of the regression task and teh second value the upper limit.
            regression_reference: A reference value which is inside the range of expected values (best if
                it was in the middle, but does not have to). Choosing different references will result
                in different explanations.
            return_importances: Whether the importance / explanation tensors should be returned as an output
                of the model. If this is True, the output of the model will be a 3-tuple:
                (output, node importances, edge importances), otherwise it is just the output itself
        """
        KerasStudentModel.__init__(self, name=name, variant=variant, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_edge_features = use_edge_features
        self.importance_units = importance_units
        self.importance_channels = importance_channels
        self.importance_activation = importance_activation
        self.importance_dropout_rate = importance_dropout_rate
        self.importance_factor = importance_factor
        self.importance_multiplier = importance_multiplier
        self.sparsity_factor = sparsity_factor
        self.concat_heads = concat_heads
        self.final_units = final_units
        self.final_dropout_rate = final_dropout_rate
        self.final_activation = final_activation
        self.final_pooling = final_pooling
        self.regression_limits = regression_limits
        self.regression_reference = regression_reference
        self.return_importances = return_importances

        # ~ MAIN CONVOLUTIONAL / ATTENTION LAYERS
        self.attention_layers: t.List[GraphBaseLayer] = []
        for u in self.units:
            lay = MultiHeadGATV2Layer(
                units=u,
                num_heads=self.importance_channels,
                use_edge_features=self.use_edge_features,
                activation=self.activation,
                use_bias=self.use_bias,
                has_self_loops=True,
                concat_heads=self.concat_heads
            )
            self.attention_layers.append(lay)

        self.lay_dropout = DropoutEmbedding(rate=self.dropout_rate)
        self.lay_sparsity = ExplanationSparsityRegularization(factor=self.sparsity_factor)

        # ~ EDGE IMPORTANCES
        self.lay_act_importance = ActivationEmbedding(activation=self.importance_activation)
        self.lay_concat_alphas = LazyConcatenate(axis=-1)

        self.lay_pool_edges_in = PoolingLocalEdges(pooling_method='mean', pooling_index=0)
        self.lay_pool_edges_out = PoolingLocalEdges(pooling_method='mean', pooling_index=1)
        self.lay_average = LazyAverage()

        # ~ NODE IMPORTANCES
        self.node_importance_units = importance_units + [self.importance_channels]
        self.node_importance_acts = ['relu' for _ in importance_units] + ['linear']
        self.node_importance_layers = []
        for u, act in zip(self.node_importance_units, self.node_importance_acts):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=use_bias
            )
            self.node_importance_layers.append(lay)

        # ~ OUTPUT / MLP TAIL END
        self.lay_pool_out = PoolingNodes(pooling_method=self.final_pooling)
        self.lay_concat_out = LazyConcatenate(axis=-1)
        self.lay_final_dropout = DropoutEmbedding(rate=self.final_dropout_rate)

        self.final_acts = ["relu" for _ in self.final_units]
        self.final_acts[-1] = self.final_activation
        self.final_biases = [True for _ in self.final_units]
        self.final_biases[-1] = False
        self.final_layers = []
        for u, act, bias in zip(self.final_units, self.final_acts, self.final_biases):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=use_bias
            )
            self.final_layers.append(lay)

        # ~ EXPLANATION ONLY TRAIN STEP
        self.bce_loss = ks.losses.BinaryCrossentropy()
        self.compiled_classification_loss = compile_utils.LossesContainer(self.bce_loss)

        self.mse_loss = ks.losses.MeanSquaredError()
        self.mae_loss = ks.losses.MeanAbsoluteError()
        self.compiled_regression_loss = compile_utils.LossesContainer(self.mae_loss)

        if self.regression_limits is not None:
            self.regression_width = np.abs(self.regression_limits[1] - self.regression_limits[0])

    def get_config(self):
        config = super(MeganStudent, self).get_config()
        config.update({
            "name": self.name,
            "variant": self.variant,
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "use_edge_features": self.use_edge_features,
            "importance_units": self.importance_units,
            "importance_channels": self.importance_channels,
            "importance_activation": self.importance_activation,
            "importance_dropout_rate": self.importance_dropout_rate,
            "importance_factor": self.importance_factor,
            "importance_multiplier": self.importance_multiplier,
            "sparsity_factor": self.sparsity_factor,
            "concat_heads": self.concat_heads,
            "final_units": self.final_units,
            "final_dropout_rate": self.final_dropout_rate,
            "final_activation": self.final_activation,
            "final_pooling": self.final_pooling,
            "regression_limits": self.regression_limits,
            "regression_reference": self.regression_reference,
            "return_importances": self.return_importances
        })

        return config
    @property
    def doing_regression(self) -> bool:
        return self.regression_limits is not None

    def call(self,
             inputs,
             training: bool = False,
             return_importances: bool = False):

        node_input, edge_input, edge_index_input = inputs

        # First of all we apply all the graph convolutional / attention layers. Each of those layers outputs
        # the attention logits alpha additional to the node embeddings. We collect all the attention logits
        # in a list so that we can later sum them all up.
        alphas = []
        x = node_input
        for lay in self.attention_layers:
            # x: ([batch], [N], F)
            # alpha: ([batch], [M], K, 1)
            x, alpha = lay([x, edge_input, edge_index_input])
            if training:
                x = self.lay_dropout(x, training=training)

            alphas.append(alpha)

        # We sum up all the individual layers attention logit tensors and the edge importances are directly
        # calculated by applying a sigmoid on that sum.
        alphas = self.lay_concat_alphas(alphas)
        edge_importances = tf.reduce_sum(alphas, axis=-1, keepdims=False)
        edge_importances = self.lay_act_importance(edge_importances)

        # Part of the final node importance tensor is actually the pooled edge importances, so that is what
        # we are doing here. The caveat here is that we assume undirected edges as two directed edges in
        # opposing direction. To now achieve a symmetric pooling of these edges we have to pool in both
        # directions and then use the average of both.
        pooled_edges_in = self.lay_pool_edges_in([node_input, edge_importances, edge_index_input])
        pooled_edges_out = self.lay_pool_edges_out([node_input, edge_importances, edge_index_input])
        pooled_edges = self.lay_average([pooled_edges_out, pooled_edges_in])

        node_importances_tilde = x
        for lay in self.node_importance_layers:
            node_importances_tilde = lay(node_importances_tilde)

        node_importances_tilde = self.lay_act_importance(node_importances_tilde)

        node_importances = node_importances_tilde * pooled_edges
        self.lay_sparsity(node_importances)

        # Here we apply the global pooling. It is important to note that we do K separate pooling operations
        # were each time we use the same node embeddings x but a different slice of the node importances as
        # the weights! We concatenate all the individual results in the end.
        outs = []
        for k in range(self.importance_channels):
            node_importance_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)
            out = self.lay_pool_out(x * node_importance_slice)

            outs.append(out)

        # out: ([batch], F*K)
        out = self.lay_concat_out(outs)

        # Now "out" is a graph embedding vector of known dimension so we can simply apply the normal dense
        # mlp to get the final output value.
        for lay in self.final_layers:
            out = lay(out)
            if training:
                self.lay_final_dropout(out, training=training)

        out = out # + self.regression_reference
        # Usually, the node and edge importance tensors would be direct outputs of the model as well, but
        # we need the option to just return the output alone to be compatible with the standard model
        # evaluation pipeline already implemented in the library.
        if self.return_importances or return_importances:
            return out, node_importances, edge_importances
        else:
            return out

    def regression_augmentation(self,
                                out_true):
        center_distances = tf.abs(out_true - self.regression_reference)
        center_distances = (center_distances * self.importance_multiplier) / (0.5 * self.regression_width)
        #center_distances = tf.expand_dims(center_distances, axis=-1)

        # So we need two things: a "samples" tensor and a "mask" tensor. We are going to use the samples
        # tensor as the actual ground truth which acts as the regression target during the explanation
        # train step. The binary values of the mask will determine at which positions a loss should
        # actually be calculated for both of the channels

        # The "lower" part is all the samples which have a target value below the reference value.
        lo_mask = tf.where(out_true < self.regression_reference, 1.0, 0.0)

        # The "higher" part all of the samples above reference
        hi_mask = tf.where(out_true > self.regression_reference, 1.0, 0.0)

        samples = tf.concat([center_distances, center_distances], axis=-1)
        mask = tf.concat([lo_mask, hi_mask], axis=-1)

        return samples, mask

    def train_step_explanation(self, x, y,
                               update_weights: bool = True):

        if self.return_importances:
            out_true, _, _ = y
        else:
            out_true = y

        exp_loss = 0
        with tf.GradientTape() as tape:

            y_pred = self(x, training=True, return_importances=True)
            out_pred, ni_pred, ei_pred = y_pred

            # First of all we need to assemble the approximated model output, which is simply calculated
            # by applying a global pooling operation on the corresponding slice of the node importances.
            # So for each slice (each importance channel) we get a single value, which we then concatenate
            # into an output vector with K dimensions.
            outs = []
            for k in range(self.importance_channels):
                node_importances_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)
                out = self.lay_pool_out(node_importances_slice)

                outs.append(out)

            # outs: ([batch], K)
            outs = self.lay_concat_out(outs)

            if self.doing_regression:
                out_true, mask = self.regression_augmentation(out_true)
                out_pred = outs
                exp_loss = self.compiled_mse_loss(out_true * mask, out_pred * mask)

            else:
                #out_pred = ks.backend.sigmoid(outs)
                out_pred = shifted_sigmoid(outs)
                exp_loss = self.compiled_bce_loss(out_true, out_pred)

            exp_loss *= self.importance_factor

        # Compute gradients
        trainable_vars = self.trainable_variables
        exp_gradients = tape.gradient(exp_loss, trainable_vars)

        # Update weights
        if update_weights:
            self.optimizer.apply_gradients(zip(exp_gradients, trainable_vars))

        return {'exp_loss': exp_loss}

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # The "train_step_explanation" method will execute the entire explanation train step once. This
        # mainly involves creating an approximate solution of the original regression / classification
        # problem using ONLY the node importances of the different channels and then performing one
        # complete weight update based on the corresponding loss.
        if self.importance_factor != 0 and False:
            exp_metrics = self.train_step_explanation(x, y, False)
        else:
            exp_metrics = {}

        exp_loss = 0
        with tf.GradientTape() as tape:

            out_true, ni_true, ei_true = y
            out_pred, ni_pred, ei_pred = self(x, training=True, return_importances=True)
            loss = self.compiled_loss(
                [out_true, ni_true, ei_true],
                [out_pred, ni_pred, ei_pred],
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            if self.importance_factor != 0:
                # ~ explanation loss
                # First of all we need to assemble the approximated model output, which is simply calculated
                # by applying a global pooling operation on the corresponding slice of the node importances.
                # So for each slice (each importance channel) we get a single value, which we then
                # concatenate into an output vector with K dimensions.
                outs = []
                for k in range(self.importance_channels):
                    node_importances_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)
                    out = self.lay_pool_out(node_importances_slice)

                    outs.append(out)

                # outs: ([batch], K)
                outs = self.lay_concat_out(outs)

                if self.doing_regression:
                    out_true, mask = self.regression_augmentation(out_true)
                    out_pred = outs
                    exp_loss = self.importance_channels * self.compiled_regression_loss(out_true * mask,
                                                                                        out_pred * mask)

                else:
                    # out_pred = ks.backend.sigmoid(outs)
                    out_pred = shifted_sigmoid(outs)
                    exp_loss = self.compiled_classification_loss(out_true, out_pred)

                exp_loss *= self.importance_factor
                loss += exp_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(
            y,
            out_pred if not self.return_importances else [out_pred, ni_pred, ei_pred],
            sample_weight=sample_weight
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            **{m.name: m.result() for m in self.metrics},
            **exp_metrics
        }
