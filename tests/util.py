import os
import sys
import pathlib
import logging
import random
import typing as t

import numpy as np
from decouple import config

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

LOG_TESTING = config('LOG_TESTING', cast=bool, default=True)
LOG = logging.getLogger('Testing')
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.NullHandler())
if LOG_TESTING:
    LOG.addHandler(logging.StreamHandler(sys.stdout))


def generate_mock_dataset_keras(num_samples: int,
                                num_node_attributes: int,
                                num_edge_attributes: int,
                                num_outputs: int,
                                num_importance_channels: t.Optional[int] = None,
                                num_node_limits: t.Tuple[int, int] = (10, 20)
                                ) -> t.List[dict]:

    dataset = []
    for s in range(num_samples):
        g = {}
        # First thing to generate the graph is to determine the number of nodes which the graph is
        # supposed to have and then create an array of node indices.
        # We can also generate random features for each node already
        num_nodes = random.randint(*num_node_limits)
        node_indices = np.arange(0, num_nodes, 1, dtype=int)
        node_attributes = np.random.rand(num_nodes, num_node_attributes)
        g['node_indices'] = node_indices
        g['node_attributes'] = node_attributes
        g['graph_labels'] = np.random.rand(num_outputs)

        # The actual graph generation works like this: We start with a seed node (0) and then maintain a
        # list of "inserted" nodes and one with "not inserted" nodes and then in each iteration we insert
        # an edge between two elements of the two sets
        remaining = list(node_indices.tolist())
        inserted = [remaining.pop(0)]

        edge_indices = []
        while len(remaining) != 0:
            node_in = random.choice(inserted)
            node_out = random.choice(remaining)

            edge_indices.append([node_in, node_out])
            edge_indices.append([node_out, node_in])

            remaining.remove(node_out)

        num_edges = len(edge_indices)
        edge_indices = np.array(edge_indices, dtype=int)
        edge_attributes = np.random.rand(num_edges, num_edge_attributes)
        g['edge_indices'] = edge_indices
        g['edge_attributes'] = edge_attributes

        if num_importance_channels is not None:
            node_importances = np.random.rand(num_nodes, num_importance_channels)
            edge_importances = np.random.rand(num_edges, num_importance_channels)
            g['node_importances'] = node_importances
            g['edge_importances'] = edge_importances

        dataset.append(g)

    return dataset
