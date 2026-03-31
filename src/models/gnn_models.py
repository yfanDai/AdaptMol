import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn.inits import glorot, zeros
# from message_passing_for_gat import MessagePassingForGAT

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x, edge_attr=edge_embeddings, node_dim=-3)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 4:
            x, edge_index, edge_attr, atom_mask_index = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise NotImplementedError(f'error value in {self.JK}')

        if len(argv) == 4:
            graph_representation = scatter_mean(node_representation, atom_mask_index, dim=0)
            return node_representation, graph_representation
        else:
            return node_representation

class CrossAttention(nn.Module):
    def __init__(self, emb_dim, attr_dim, heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        
        # 线性层将输入投影为 Query, Key, Value
        self.query_proj = nn.Linear(emb_dim, emb_dim)   # 输入: graph_repr
        self.key_proj = nn.Linear(attr_dim, emb_dim)    # 输入: attrs
        self.value_proj = nn.Linear(attr_dim, emb_dim)  # 输入: attrs
        
        # 多头注意力层 (PyTorch 内置)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, 
                                                    num_heads=heads, 
                                                    dropout=dropout, 
                                                    batch_first=True)
        
        # 用于将多头注意力的输出映射回 emb_dim 维度
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, graph_repr, attrs):
        """
        graph_repr: [B, emb_dim] (节点表示)
        attrs: [B,  attr_dim] (节点的属性表示)
        """

        # 1. 线性变换生成 Query, Key, Value
        query = self.query_proj(graph_repr).unsqueeze(1)  # [B, 1, emb_dim]
        key = self.key_proj(attrs).unsqueeze(1)  # [B, num_attr, emb_dim]
        value = self.value_proj(attrs).unsqueeze(1)  # [B, num_attr, emb_dim]

        # 2. 多头注意力计算
        attn_output, attn_weights = self.multihead_attn(query, key, value)  # [B, 1, emb_dim], [B, 1, num_attr]

        # 3. 输出投影并去掉维度 1
        attn_output = self.output_proj(attn_output).squeeze(1)  # [B, emb_dim]

        return attn_output, attn_weights

        
# class attributes_GNN(torch.nn.Module):
#     def __init__(self, num_layer, emb_dim, attr_dim, device, JK="last", drop_ratio=0, gnn_type="gin",with_attr=True,pretrained_bool='False'):
#         super(attributes_GNN, self).__init__()

#         self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

#         self.node_fc = nn.Linear(emb_dim + attr_dim, 1)

#         self.dim_fc = nn.Linear(emb_dim+attr_dim, emb_dim)

#         self.with_attr = with_attr

#         if pretrained_bool == "True":
#             pretrained_path = './model_gin/{}_supervised_contextpred.pth'.format(gnn_type)
#             self.gnn.load_state_dict(torch.load(pretrained_path, map_location=device))


#     def forward(self, *argv):
#         x, edge_index, edge_attr, atom_mask_index, attrs = argv[0], argv[1], argv[2], argv[3], argv[4]
#         node_representation, graph_representation = self.gnn(x, edge_index, edge_attr, atom_mask_index)
#         # print("-------------test---------------\n")
#         # print(graph_representation.shape)
#         # print(node_representation.shape)
#         # print(attrs.shape)

#         if self.with_attr == "False":
#             # print("I an here\n")
#             return graph_representation

#         node_attrs = attrs[atom_mask_index]
#         node_concat_attrs = torch.cat([node_representation, node_attrs], dim=1)

#         node_weights = torch.sigmoid(self.node_fc(node_concat_attrs.float()))

#         node_representation = node_representation * node_weights

#         graph_representation = scatter_mean(node_representation, atom_mask_index, dim=0)

#         graph_concat_attrs = torch.cat([graph_representation, attrs], dim=1)
#         dim_weights = torch.sigmoid(self.dim_fc(graph_concat_attrs.float()))
#         graph_representation = graph_representation * dim_weights


#         return graph_representation

# class attributes_GNN(torch.nn.Module):
#     def __init__(self, num_layer, emb_dim, attr_dim, device, JK="last", 
#                  drop_ratio=0, gnn_type="gin", with_attr=True, pretrained_bool='False'):
#         super(attributes_GNN, self).__init__()

#         self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

#         # 使用 Cross-Attention 代替原来的 FC 层
#         self.node_attention = CrossAttention(emb_dim, attr_dim, heads=4, dropout=0.1)
#         self.dim_fc = nn.Linear(emb_dim + attr_dim, emb_dim)

#         self.with_attr = with_attr

#         if pretrained_bool == "True":
#             pretrained_path = f'./model_gin/{gnn_type}_supervised_contextpred.pth'
#             self.gnn.load_state_dict(torch.load(pretrained_path, map_location=device))

#     def forward(self, *argv):
#         x, edge_index, edge_attr, atom_mask_index, attrs = argv

#         # 获取节点和图的表示
#         node_representation, graph_representation = self.gnn(x, edge_index, edge_attr, atom_mask_index)

#         if self.with_attr == "False":
#             return graph_representation

#         # 根据 atom_mask_index 获取对应的节点属性
#         node_attrs = attrs[atom_mask_index]
#         # 使用 Cross-Attention 计算节点的加权表示
#         # print("-------test---------\n")
#         # print(node_attrs.shape)
#         # print(node_representation.shape)
#         node_representation, attn_weights = self.node_attention(node_representation, node_attrs)

#         # 聚合节点表示为图的表示
#         graph_representation = scatter_mean(node_representation, atom_mask_index, dim=0)

#         graph_concat_attrs = torch.cat([graph_representation, attrs], dim=1)
#         dim_weights = torch.sigmoid(self.dim_fc(graph_concat_attrs.float()))
#         graph_representation = graph_representation * dim_weights

#         return graph_representation, attn_weights
class attributes_GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, attr_dim, device, JK="last", drop_ratio=0, gnn_type="gin", with_attr=True, pretrained_bool='False',dropout = 0.0 , task_num = 1):
        super(attributes_GNN, self).__init__()

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # 使用CrossAttention替换FC
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim + attr_dim, num_heads=8)

        self.dim_fc = nn.Linear(emb_dim + 2*attr_dim, emb_dim + attr_dim)

        self.with_attr = with_attr

        if pretrained_bool == "True":
            pretrained_path = './model_gin/{}_supervised_contextpred.pth'.format(gnn_type)
            self.gnn.load_state_dict(torch.load(pretrained_path, map_location=device))

        self.ffn = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.Linear(in_features=emb_dim + attr_dim, out_features=(emb_dim + attr_dim)//2, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(in_features=(emb_dim + attr_dim)//2, out_features=task_num, bias=True)
                                    )

    def forward(self, *argv):
        x, edge_index, edge_attr, atom_mask_index, attrs = argv[0], argv[1], argv[2], argv[3], argv[4]
        node_representation, graph_representation = self.gnn(x, edge_index, edge_attr, atom_mask_index)

        if not self.with_attr:
            return graph_representation
        device = torch.device("cuda:1")
        attrs = attrs.to(device)
        # print("---------hope test final------\n")
        # print(attrs)
        # print(atom_mask_index)
        node_attrs = attrs[atom_mask_index]
        node_concat_attrs = torch.cat([node_representation, node_attrs], dim=1)

        # 使用Attention
        node_concat_attrs = node_concat_attrs.unsqueeze(1)  # 添加序列维度
        attn_output, attn_weights = self.attention(node_concat_attrs, node_concat_attrs, node_concat_attrs)
        node_representation = attn_output.squeeze(1)  # 去掉序列维度

        graph_representation = scatter_mean(node_representation, atom_mask_index, dim=0)

        graph_concat_attrs = torch.cat([graph_representation, attrs], dim=1)
        # print("--------test-----------\n")
        # print(graph_representation.shape)

        dim_weights = torch.sigmoid(self.dim_fc(graph_concat_attrs.float()))
        graph_representation = graph_representation * dim_weights

        output = self.ffn(graph_representation)

        return output