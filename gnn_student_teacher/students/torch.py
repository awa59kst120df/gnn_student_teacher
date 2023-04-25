
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

from gnn_student_teacher.students import AbstractStudentModel


class TorchStudentModel(AbstractStudentModel, torch.nn.Module):

    def __init__(self, name, variant, **kwargs):
        AbstractStudentModel.__init__(self, name, variant)
        torch.nn.Module.__init__(self)
        self.name = self.full_name


class MockTorchStudent(TorchStudentModel):

    def __init__(self,
                 name: str,
                 variant: str,
                 input_units: int,
                 hidden_units: int,
                 output_units: int):
        TorchStudentModel.__init__(self, name, variant)
        self.lay_conv = GCNConv(input_units, hidden_units)
        self.lay_lin = Linear(hidden_units, output_units)

    def forward(self, node_input, edge_input, edge_index, batch):

        x = self.lay_conv(node_input, edge_index, edge_weight=edge_input)
        x = global_add_pool(x, batch)
        x = self.lay_lin(x)

        return x
