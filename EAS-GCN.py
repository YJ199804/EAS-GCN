import argparse
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from train_eval import *
from datasets import *

import warnings


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--dropout_rate', type=float, default=0.8)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--p', type=float, default=0.5)
parser.add_argument('--use_mlp', type=bool, default=False)
parser.add_argument('--n_enc_1', type=int, default=600)
parser.add_argument('--n_dec_1', type=int, default=600)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--mlp_enc_1', type=int, default=100)
parser.add_argument('--mlp_enc_2', type=int, default=100)
args = parser.parse_args()

# Auto-Encoder
class AE(nn.Module):

    def __init__(self, n_input):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, args.n_enc_1)
        self.class_layer = Linear(args.n_enc_1, args.n_hidden)

        self.dec_1 = Linear(args.n_hidden, args.n_dec_1)
        self.x_bar_layer = Linear(args.n_dec_1, n_input)

    def reset_parameters(self):
        self.enc_1.reset_parameters()
        self.class_layer.reset_parameters()
        self.dec_1.reset_parameters()
        self.x_bar_layer.reset_parameters()

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        z = self.class_layer(enc_h1)

        dec_h1 = F.relu(self.dec_1(z))
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, enc_h1, z


class EAS_GCN(nn.Module):

    def __init__(self, n_input, n_class):
        super(EAS_GCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(n_input=n_input)
        #         self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # MLP
        self.lin1 = Linear(n_input, args.mlp_enc_1)
        self.lin2 = Linear(args.mlp_enc_1, args.mlp_enc_2)

        # GCN for inter information
        #         self.gnn_1 = GCNConv(mlp_enc_2, n_enc_1, improved=True)
        self.gnn_1_mlp = GCNConv(args.mlp_enc_2, args.n_enc_1)
        self.gnn_1 = GCNConv(n_input, args.n_enc_1)
        self.gnn_2 = GCNConv(args.n_enc_1, n_class)

        self.pred_node_degree = GCNConv(args.n_hidden, 1)

    def reset_parameters(self):
        self.ae.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.gnn_1.reset_parameters()
        self.gnn_1_mlp.reset_parameters()
        self.gnn_2.reset_parameters()
        self.pred_node_degree.reset_parameters()

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, z = self.ae(x)


        # MLP
        if args.use_mlp == True:
            x = F.dropout(x, p=args.dropout_rate, training=self.training)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=args.dropout_rate, training=self.training)
            x = self.lin2(x)
            h = self.gnn_1_mlp(x, adj)
            h = self.gnn_2((1 - args.p) * h + args.p * tra1, adj)
            h_pred_nd = self.pred_node_degree(z, adj)
            predict = F.log_softmax(h, dim=1)
        else:
            # GCN Module
            h = self.gnn_1(x, adj)
            h = self.gnn_2((1 - args.p) * h + args.p * tra1, adj)
            h_pred_nd = self.pred_node_degree(z, adj)
            predict = F.log_softmax(h, dim=1)
        #         predict = F.softmax(h, dim=1)

        return x_bar, predict, h_pred_nd


warnings.filterwarnings("ignore", category=UserWarning)

if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset[0])
    run(dataset, EAS_GCN(dataset.num_features, dataset.num_classes), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha, args.beta, permute_masks,
        lcc=False)
elif args.dataset == "cs" or args.dataset == "physics":
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])
    run(dataset, EAS_GCN(dataset.num_features, dataset.num_classes), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha, args.beta, permute_masks,
        lcc=False)
elif args.dataset == "computers" or args.dataset == "photo":
    dataset = get_amazon_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])
    run(dataset, EAS_GCN(dataset.num_features, dataset.num_classes), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha, args.beta, permute_masks,
        lcc=True)