import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
import numpy as np

from collections import deque

import math

_CAYLEY_BOUNDS = [
    (6, 2),
    (24, 3),
    (120, 5),
    (336, 7),
    (1320, 11),
    (2184, 13),
    (4896, 17),
    (6840, 19),
    (12144, 23),
    (24360, 29),
    (29760, 31),
    (50616, 37),
]

def build_cayley_bank():

    ret_edges = []

    for _, p in _CAYLEY_BOUNDS:
        generators = np.array([
            [[1, 1], [0, 1]],
            [[1, p-1], [0, 1]],
            [[1, 0], [1, 1]],
            [[1, 0], [p-1, 1]]])
        ind = 1

        queue = deque([np.array([[1, 0], [0, 1]])])
        nodes = {(1, 0, 0, 1): 0}

        senders = []
        receivers = []

        while queue:
            x = queue.pop()
            x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
            assert x_flat in nodes
            ind_x = nodes[x_flat]
            for i in range(4):
                tx = np.matmul(x, generators[i])
                tx = np.mod(tx, p)
                tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
                if tx_flat not in nodes:
                    nodes[tx_flat] = ind
                    ind += 1
                    queue.append(tx)
                ind_tx = nodes[tx_flat]

                senders.append(ind_x)
                receivers.append(ind_tx)

        ret_edges.append((p, [senders, receivers]))

    return ret_edges

def batched_augment_cayley(num_graphs, batch, cayley_bank):
    node_lims = np.zeros(num_graphs)
    mappings = [[] for _ in range(num_graphs)]
    fake_mappings = [[] for _ in range(num_graphs)]

    for i in range(len(batch)):
        node_lims[batch[i]] += 1
        mappings[batch[i]].append(i)
        fake_mappings[batch[i]].append(i)

    senders = []
    receivers = []

    fake_senders = []
    fake_receivers = []

    og_index = len(batch)
    start_index = len(batch)

    for g in range(num_graphs):
        p = 2
        chosen_i = -1
        for i in range(len(_CAYLEY_BOUNDS)):
            sz, p = _CAYLEY_BOUNDS[i]
            if sz >= node_lims[g]:
                chosen_i = i
                break

        assert chosen_i >= 0
        _p, edge_pack = cayley_bank[chosen_i]
        assert p == _p

        if sz > int(node_lims[g]):
            missing_nodes = list(range(start_index, start_index + int(sz - node_lims[g])))
            fake_mappings[g].extend(missing_nodes)
            start_index += len(missing_nodes)

        r_mappings = mappings[g]
        f_mappings = fake_mappings[g]

        assert sz == len(f_mappings)

        for v, w in zip(*edge_pack):
            if v < node_lims[g] and w < node_lims[g]:
                senders.append(r_mappings[v])
                receivers.append(r_mappings[w])
            else:
                fake_senders.append(f_mappings[v])
                fake_receivers.append(f_mappings[w])

    self_indexes_senders = []
    self_indexes_recievers = []
    for i in range(og_index, start_index):
        self_indexes_senders.append(i)
        self_indexes_recievers.append(i)

    senders.extend(fake_senders)
    receivers.extend(fake_receivers)

    for i in range(start_index):
        senders.append(i)
        receivers.append(i)

    edge_attr = []
    for i in range(len(senders)):
        edge_attr.append([0, 0, 0])  # Replace this with whatever edge features would be expected in your dataset :)

    return [senders, receivers], edge_attr, start_index, [self_indexes_senders, self_indexes_recievers]

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.cayley_bank = build_cayley_bank()

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        num_graphs = batched_data.num_graphs
        num_nodes = x.shape[0]

        # self_edge_index are just the self connections, outside of the graph node range.
        # I.e. if it ends at 100, then [101, 101]...
        # From the batched Cayley graph this is outside of the range
        cayley_g, cayley_attr, max_node, self_edge_indexes = batched_augment_cayley(num_graphs, batch, self.cayley_bank)
        cayley_g = torch.LongTensor(cayley_g).cuda()
        cayley_attr = torch.LongTensor(cayley_attr).cuda()

        # New self edges for dummy nodes
        self_edge_indexes = torch.LongTensor(self_edge_indexes).cuda()
        # edges indexes to be be used that include real edges for the graph + self edges for dummy nodes
        new_edge_index = torch.cat((edge_index, self_edge_indexes), dim=1).cuda()

        h_list = [self.atom_encoder(x)]

        # Initialise them at layer 0 <- I just took the average nodes at this point
        h_list_zero = h_list[0]

        # Example of different strategies to initialise the dummy nodes
        # Create the dummy nodes, expanding on layer[0] average
        average_real_nodes = torch.mean(h_list[0], dim=0).cuda()
        dummy_nodes = average_real_nodes.expand(max_node - num_nodes, -1).cuda()

        # I have changed this to -1, 0
        # dummy_nodes = torch.randn(max_node - num_nodes, self.emb_dim).to(device)

        # Expand the dummy edge attr to match new_edge_index shape, of course we're just adding [0, 0, 0]
        # Set the first nodes to what they were + [0, 0, 0] for everything else
        dummy_edge_attr = torch.zeros(new_edge_index.shape[1], 3, dtype=torch.int64).cuda()
        dummy_edge_attr[:edge_attr.shape[0], :] = edge_attr

        # Expand the first layer to include dummy_nodes
        h_list[0] = torch.cat((h_list[0], dummy_nodes), dim=0)

        for layer in range(self.num_layer):
            if layer % 2 == 1:
                h = self.convs[layer](h_list[layer], cayley_g, cayley_attr)
            else:
                h = self.convs[layer](h_list[layer], new_edge_index, dummy_edge_attr)

            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        node_representation = node_representation[:num_nodes]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
