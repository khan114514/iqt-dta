import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict


'''
MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction
'''


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num, hidden_dim=96, out_dim=96):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_num, hidden_dim, 3)
            )

        self.linear = nn.Linear(block_num * hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data
    
    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)


    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config = (3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x


class InteractionPriorModule(nn.Module):
    """
    Extension point for paper-specific interaction priors.
    The current implementation learns a shared interaction representation from
    protein and ligand embeddings without changing the backbone encoders.
    """

    def __init__(self, protein_dim, ligand_dim, hidden_dim):
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, protein_x, ligand_x):
        protein_h = self.protein_proj(protein_x)
        ligand_h = self.ligand_proj(ligand_x)
        return self.out_norm(torch.sigmoid(protein_h * ligand_h))


class QuantityBranchModule(nn.Module):
    """
    Extension point for a quantity-aware auxiliary branch.
    It emits a latent branch feature used by the final regressor plus an
    optional auxiliary scalar prediction when a separate quantity target is
    available in the batch.
    """

    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fused_x):
        hidden = self.encoder(fused_x)
        pred = self.head(hidden)
        return hidden, pred


class DecorrelationRegularizer(nn.Module):
    """
    Penalizes squared cosine similarity between two representations after
    projecting them into a shared space. The caller decides which pair of
    tensors should be decorrelated.
    """

    def __init__(self, first_dim, second_dim, hidden_dim):
        super().__init__()
        self.first_proj = nn.Linear(first_dim, hidden_dim)
        self.second_proj = nn.Linear(second_dim, hidden_dim)

    def forward(self, first_x, second_x):
        first_h = self.first_proj(first_x)
        second_h = self.second_proj(second_x)
        first_h = first_h - first_h.mean(dim=0, keepdim=True)
        second_h = second_h - second_h.mean(dim=0, keepdim=True)
        first_h = F.normalize(first_h, p=2, dim=-1)
        second_h = F.normalize(second_h, p=2, dim=-1)
        return (first_h * second_h).sum(dim=-1).pow(2).mean()

class MGraphDTA(nn.Module):
    def __init__(
        self,
        block_num,
        vocab_protein_size,
        embedding_size=128,
        filter_num=32,
        out_dim=1,
        protein_hidden_dim=96,
        protein_out_dim=96,
        ligand_block_config=(8, 8, 8),
        ligand_bn_sizes=(2, 2, 2),
        ablate_protein=False,
        ablate_ligand=False,
        classifier_hidden_dims=(1024, 1024, 256),
        dropout=0.1,
        use_interaction_prior=False,
        use_quantity_branch=False,
        use_decorrelation_regularizer=False,
        interaction_prior_dim=128,
        quantity_branch_dim=128,
        decorrelation_dim=128,
        interaction_prior_weight=1.0,
        quantity_branch_weight=1.0,
        quantity_loss_weight=0.2,
        decorrelation_loss_weight=1e-2,
    ):
        super().__init__()
        self.ablate_protein = ablate_protein
        self.ablate_ligand = ablate_ligand
        self.protein_out_dim = protein_out_dim
        self.ligand_out_dim = filter_num * 3
        self.base_fusion_dim = self.protein_out_dim + self.ligand_out_dim
        self.use_interaction_prior = use_interaction_prior
        self.use_quantity_branch = use_quantity_branch
        self.use_decorrelation_regularizer = use_decorrelation_regularizer
        self.interaction_prior_weight = interaction_prior_weight
        self.quantity_branch_weight = quantity_branch_weight
        self.quantity_loss_weight = quantity_loss_weight
        self.decorrelation_loss_weight = decorrelation_loss_weight

        self.protein_encoder = TargetRepresentation(
            block_num,
            vocab_protein_size,
            embedding_size,
            hidden_dim=protein_hidden_dim,
            out_dim=protein_out_dim,
        )
        self.ligand_encoder = GraphDenseNet(
            num_input_features=22,
            out_dim=self.ligand_out_dim,
            block_config=list(ligand_block_config),
            bn_sizes=list(ligand_bn_sizes),
        )

        fusion_input_dim = self.base_fusion_dim
        if self.use_interaction_prior:
            self.interaction_prior = InteractionPriorModule(
                protein_dim=self.protein_out_dim,
                ligand_dim=self.ligand_out_dim,
                hidden_dim=interaction_prior_dim,
            )
            fusion_input_dim += interaction_prior_dim
        else:
            self.interaction_prior = None

        if self.use_quantity_branch:
            self.quantity_branch = QuantityBranchModule(
                in_dim=self.base_fusion_dim,
                hidden_dim=quantity_branch_dim,
                out_dim=out_dim,
                dropout=dropout,
            )
            fusion_input_dim += quantity_branch_dim
        else:
            self.quantity_branch = None

        if self.use_decorrelation_regularizer:
            if self.use_interaction_prior and self.use_quantity_branch:
                self.decorrelation_mode = "branch_pair"
                decorrelation_first_dim = interaction_prior_dim
                decorrelation_second_dim = quantity_branch_dim
            else:
                self.decorrelation_mode = "encoder_pair"
                decorrelation_first_dim = self.protein_out_dim
                decorrelation_second_dim = self.ligand_out_dim
            self.decorrelation_regularizer = DecorrelationRegularizer(
                first_dim=decorrelation_first_dim,
                second_dim=decorrelation_second_dim,
                hidden_dim=decorrelation_dim,
            )
        else:
            self.decorrelation_mode = None
            self.decorrelation_regularizer = None

        layers = []
        classifier_dims = [fusion_input_dim] + list(classifier_hidden_dims) + [out_dim]
        for idx in range(len(classifier_dims) - 1):
            layers.append(nn.Linear(classifier_dims[idx], classifier_dims[idx + 1]))
            if idx < len(classifier_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.classifier = nn.Sequential(*layers)

    def forward(self, data):
        target = data.target
        protein_x = self.protein_encoder(target)
        ligand_x = self.ligand_encoder(data)
        if self.ablate_protein:
            protein_x = torch.zeros_like(protein_x)
        if self.ablate_ligand:
            ligand_x = torch.zeros_like(ligand_x)

        base_fused_x = torch.cat([protein_x, ligand_x], dim=-1)
        fused_features = [protein_x, ligand_x]
        aux_outputs = {}
        aux_losses = {}
        interaction_features = None
        quantity_hidden = None

        if self.interaction_prior is not None:
            interaction_features = self.interaction_prior(protein_x, ligand_x)
            fused_features.append(interaction_features * self.interaction_prior_weight)
            aux_outputs["interaction_prior"] = interaction_features

        if self.quantity_branch is not None:
            # Keep the quantity branch independent from interaction_prior so
            # the two branches can be meaningfully decorrelated.
            quantity_hidden, quantity_pred = self.quantity_branch(base_fused_x)
            fused_features.append(quantity_hidden * self.quantity_branch_weight)
            aux_outputs["quantity_hidden"] = quantity_hidden
            aux_outputs["quantity_prediction"] = quantity_pred

        if self.decorrelation_regularizer is not None:
            if self.decorrelation_mode == "branch_pair" and interaction_features is not None and quantity_hidden is not None:
                decorrelation_value = self.decorrelation_regularizer(interaction_features, quantity_hidden)
            else:
                decorrelation_value = self.decorrelation_regularizer(protein_x, ligand_x)
            aux_losses["decorrelation_regularizer"] = decorrelation_value * self.decorrelation_loss_weight

        x = torch.cat(fused_features, dim=-1)

        x = self.classifier(x)

        return {
            "prediction": x,
            "aux_outputs": aux_outputs,
            "aux_losses": aux_losses,
        }
