import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_student_teacher.students import AbstractStudentModel
from gnn_student_teacher.students.torch import MockTorchStudent


# == MockTorchStudent ==

def test_mock_torch_student_basically_works():
    hidden_units = 10
    output_units = 5
    student = MockTorchStudent(
        'mock', 'ref',
        input_units=2,
        hidden_units=hidden_units,
        output_units=output_units,
    )
    assert isinstance(student, MockTorchStudent)
    assert isinstance(student, AbstractStudentModel)
    assert isinstance(student, torch.nn.Module)

    d1 = Data(
        x=torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.tensor([[1], [1]], dtype=torch.float)
    )
    d2 = Data(
        x=torch.tensor([[2, 2], [2, 2]], dtype=torch.float),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[1]], dtype=torch.float)
    )
    batch_size = 2
    loader = DataLoader([d1, d2], batch_size=batch_size)

    for data in loader:
        out = student(data.x, data.edge_attr, data.edge_index, data.batch)
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (batch_size, output_units)

