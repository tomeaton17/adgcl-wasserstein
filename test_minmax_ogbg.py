import argparse
import logging
import random
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

import numpy as np
import torch
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.linear_model import Ridge, LogisticRegression
from torch_geometric.data import DataLoader
from torch_geometric.utils import get_laplacian,to_scipy_sparse_matrix, to_networkx, to_undirected
from torch_geometric.transforms import Compose
from torch_scatter import scatter
import networkx as nx
import numpy.linalg as lg
import copy
import scipy.linalg as slg

from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.encoder import MoleculeEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner

def wass_dist_(A, B):
    n = len(A)
    l1_tilde = A + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    l2_tilde = B + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    s1_tilde = lg.inv(l1_tilde)
    s2_tilde = lg.inv(l2_tilde)
    Root_1= slg.sqrtm(s1_tilde)
    Root_2= slg.sqrtm(s2_tilde)
    return np.trace(s1_tilde) + np.trace(s2_tilde) - 2*np.trace(slg.sqrtm(Root_1 @ s2_tilde @ Root_1)) 


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    evaluator = Evaluator(name=args.dataset)
    print(evaluator.eval_metric)
    my_transforms = Compose([initialize_edge_weight])
    dataset = PygGraphPropPredDataset(name=args.dataset, root='./original_datasets/', transform=my_transforms)
    print("got here")

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=128, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=128, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=128, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GInfoMinMax(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                    proj_hidden_dim=args.emb_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)


    view_learner = ViewLearner(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    if 'classification' in dataset.task_type:
        ee = EmbeddingEvaluation(LogisticRegression(dual=False, fit_intercept=True, max_iter=5000),
                                 evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
                                 param_search=True)
    elif 'regression' in dataset.task_type:
        ee = EmbeddingEvaluation(Ridge(fit_intercept=True, normalize=True, copy_X=True, max_iter=5000),
                                 evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
                                 param_search=True)
    else:
        raise NotImplementedError

    model.eval()
    train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader, test_loader)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score,
                                                                                         test_score))


    model_losses = []
    view_losses = []
    view_regs = []
    view_regs_wass = []
    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):

        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)

            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            # edge_index should be the adjacency, other params can be used to reconstruct pyg graph and then plot?
            x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
            

            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr) # Not actually bernoulli parameters see paper

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias) # create scaled vector of random variables (0,1) same size and edge_logits
            # Gumbel-Max reparameterisation
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze() # edge drop probabilities [0,1]

            #indices, weights = to_undirected(batch.edge_index, batch_aug_edge_weight, reduce="min")

            #x_aug, _ = model(batch.batch, batch.x, indices, batch.edge_attr, weights)
            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)
            
            # regularization

            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()

            reg_wass = []
            counter = 0
            for i in range(batch):
                weights = torch.ones(batch[i].edge_index.size(dim=1), dtype=torch.float, device="cuda")
                ei, ew = get_laplacian(batch[i].edge_index, weights)
                l = to_scipy_sparse_matrix(ei, ew).toarray()
                stop = batch[0].edge_index.size(dim=1)
                weights_aug = batch_aug_edge_weight[counter:stop]
                counter = stop
                ei, ew = get_laplacian(batch[i].edge_index, weights_aug)
                ew = ew.detach()
                l_aug = to_scipy_sparse_matrix(ei, ew).toarray()
                reg_wass.append(wass_dist_(l, l_aug))

            reg_wass = torch.stack(reg_wass)
            reg_wass = reg.mean()
            print(reg_wass)



            # back propagation
            view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg) - (args.reg_lambda * reg_wass)
            view_loss_all += view_loss.item() * batch.num_graphs # updating augmenter params
            reg_all += reg.item()
            reg_all_wass += reg_wass.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()


            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr) # get edge bernoulli parameters


            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)
        fin_reg_wass = reg_all_wass / len(dataloader)

        logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}, Wass Reg {}'.format(epoch, fin_model_loss, fin_view_loss, fin_reg, fin_reg_wass))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)
        view_regs_wass.append(fin_reg_wass)

        model.eval()

        train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader,
                                                                     test_loader)

        logging.info(
            "Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score, val_score, test_score))

        train_curve.append(train_score)
        valid_curve.append(val_score)
        test_curve.append(test_score)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))

    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)

