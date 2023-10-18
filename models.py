import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, RGCNConv, RGATConv, HEATConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch import Tensor
from torch_geometric.nn.inits import reset


class TripleConv(MessagePassing):
    r"""
    Message passing function for directional message passing
    based on the GINE Conv operator
    Equation:
        .. math::
             x_i^{(k)} = h_\theta^{(k)}  \biggl( x_i^{(k-1)} \ +& \sum_{j \in \mathcal{N}^+(i)}
            \mathrm{ReLU}(x_i^{(k-1)}||e^{j,i}||x_j^{(k-1)}) \ +\\
            & \sum_{j \in \mathcal{N}^-(i)}
            \mathrm{ReLU}(x_j^{(k-1)}||e^{i,j}||x_i^{(k-1)}) \biggr)


    The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps

        # Whether to switch the input to the MLP based on the directionality
        self.DIRECTIONAL = True

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            in_channels = 101
            self.lin = Linear(3 * edge_dim, edge_dim)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:


        # Switch i and j based on direction of triple:
        reverse = edge_attr[:, -1] == -1

        if self.DIRECTIONAL:
            x_i[reverse], x_j[reverse] = x_j[reverse], x_i[reverse]

        #x_i[edge_attr[:, -1] == -1] = x_j[edge_attr[:, -1] == -1]
        edge_attr = edge_attr[:, :-1]


        return self.lin(torch.cat((x_i, edge_attr, x_j), 1)).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class TripleModel(torch.nn.Module):
    """
    GNN model to predict cardinality of a query,
    given the query graph and embeddings of nodes
    and edges.

    Args:


    """
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(101, 101),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(101, 101),
            # torch.nn.Dropout(p=0.2)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(101, 100),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            # torch.nn.Dropout(p=0.2)
        )

        self.conv1 = TripleConv(nn=self.mlp, edge_dim=101)
        self.conv2 = TripleConv(nn=self.mlp2, edge_dim=101)
        self.lin = Linear(200, 50)
        self.lin2 = Linear(50, 1)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_type, edge_attr, batch=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.abs(x)


class TripleModelAdapt(torch.nn.Module):
    """
    GNN model to predict cardinality of a query,
    given the query graph and embeddings of nodes
    and edges.

    Args:


    """
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(101, 101),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(101, 101),
            # torch.nn.Dropout(p=0.2)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(101, 100),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            # torch.nn.Dropout(p=0.2)
        )

        self.conv1 = TripleConv(nn=self.mlp, edge_dim=101)
        self.conv2 = TripleConv(nn=self.mlp2, edge_dim=101)
        self.lin = Linear(200, 50)
        self.lin2 = Linear(50, 1)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = global_add_pool(x, batch=batch)
        x = self.lin(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.abs(x)



class GINmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(101, 101),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(101, 101),
            # torch.nn.Dropout(p=0.2)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(101, 100),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            # torch.nn.Dropout(p=0.2)
        )

        self.conv1 = GINEConv(nn=self.mlp, edge_dim=101)
        self.conv2 = GINEConv(nn=self.mlp2, edge_dim=101)
        self.lin = Linear(200, 50)
        self.lin2 = Linear(50, 1)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_type, edge_attr, batch=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.abs(x)

