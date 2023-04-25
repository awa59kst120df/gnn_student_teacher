"""
This module actually tests the "Test utilities" from :module:``tests.util.py``
"""
import numpy as np
import random

from .util import generate_mock_dataset_keras


def test_generate_mock_dataset_keras():
    # ~ Testing the base case first
    num_samples = random.randint(3, 10)
    num_node_attributes = random.randint(2, 5)
    num_edge_attributes = random.randint(2, 5)
    num_outputs = random.randint(1, 2)

    dataset = generate_mock_dataset_keras(
        num_samples=num_samples,
        num_node_attributes=num_node_attributes,
        num_edge_attributes=num_edge_attributes,
        num_outputs=num_outputs,
    )
    assert isinstance(dataset, list)
    assert len(dataset) == num_samples
    for g in dataset:
        assert isinstance(g, dict)
        assert 'node_indices' in g
        assert 'node_attributes' in g
        assert 'edge_attributes' in g
        assert 'edge_indices' in g
        assert 'graph_labels' in g
        # In the base case, the importance fields should not be present
        assert 'node_importances' not in g
        assert 'edge_importances' not in g
        num_nodes = len(g['node_indices'])
        num_edges = len(g['edge_indices'])
        assert g['node_attributes'].shape == (num_nodes, num_node_attributes)
        assert g['edge_attributes'].shape == (num_edges, num_edge_attributes)
        assert g['graph_labels'].shape == (num_outputs, )

    # ~ We also have the option to include importance matrices in the dataset
    num_importance_channels = random.randint(1, 10)
    dataset = generate_mock_dataset_keras(
        num_samples=num_samples,
        num_node_attributes=num_node_attributes,
        num_edge_attributes=num_edge_attributes,
        num_outputs=num_outputs,
        num_importance_channels=num_importance_channels,
        num_node_limits=(30, 40)
    )
    for g in dataset:
        num_nodes = len(g['node_indices'])
        num_edges = len(g['edge_indices'])
        # We can check if the generation of the nodes works.
        assert 30 <= num_nodes <= 40

        assert 'node_importances' in g
        assert 'edge_importances' in g
        assert g['node_importances'].shape == (num_nodes, num_importance_channels)
        assert g['edge_importances'].shape == (num_edges, num_importance_channels)
