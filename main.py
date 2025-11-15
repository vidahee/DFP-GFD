import argparse
import os, copy
import random
import time, datetime
import joblib, yaml
from scipy.special import comb
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.nn.pytorch.factory import KNNGraph
from utils import load_data, prob_convert_pred, eval_metric
import sympy
import scipy
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_theta(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


class PolyConv(nn.Module):
    def __init__(self, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        return h


class BernConv(nn.Module):
    def __init__(self, orders=2):
        super().__init__()
        self.K = orders
        self.weight = nn.Parameter(torch.ones(orders + 1))

    def forward(self, graph, feat):
        def unnLaplacian1(feat, D_invsqrt, graph):
            """ \hat{L} X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        def unnLaplacian2(feat, D_invsqrt, graph):
            """ (2I - \hat{L}) X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat + graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            tmp = [feat]
            weight = nn.functional.relu(self.weight)
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            for i in range(self.K):
                feat = unnLaplacian2(feat, D_invsqrt, graph)
                tmp.append(feat)

            out_feat = (comb(self.K, 0) / (2 ** self.K)) * weight[0] * tmp[self.K]
            for i in range(self.K):
                x = tmp[self.K - i - 1]
                for j in range(i + 1):
                    x = unnLaplacian1(feat, D_invsqrt, graph)
                out_feat = out_feat + (comb(self.K, i + 1) / (2 ** self.K)) * weight[i + 1] * x
        return out_feat


class MyModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 out_feats,
                 num_layers,
                 dropout_rate,
                 activation):
        super(MyModel, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hid_feats = hid_feats
        self.num_layers = num_layers
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.thetas = calculate_theta(d=num_layers)
        self.conv_o = []
        for i in range(len(self.thetas)):
            self.conv_o.append(PolyConv(self.thetas[i]))
        # self.conv_o.append(BernConv(2))
        # for i in range(num_layers):
        #     self.conv_o.append(BernConv(2))

        self.linear1_o = nn.Linear(in_feats, hid_feats)
        self.linear2_o = nn.Linear(hid_feats, hid_feats)

        self.conv_s = []
        for i in range(len(self.thetas)):
            # self.conv_s.append(PolyConv(self.thetas[i]))
            self.conv_s.append(PolyConv(len(self.thetas[i])*[4]))
        # self.conv_s.append(BernConv(2))
        # for i in range(num_layers):
        #     self.conv_s.append(BernConv(2))

        self.linear1_s = nn.Linear(in_feats, hid_feats)
        self.linear2_s = nn.Linear(hid_feats, hid_feats)

        self.mlp = nn.Sequential(
            nn.Linear(hid_feats * len(self.conv_o) * 2, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, out_feats)).to('cuda')
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph, sim_graph):
        h = graph.ndata['x']
        h = self.linear1_o(h)
        h = self.act(h)
        # h = self.linear2_o(h)
        # h = self.act(h)
        h_final = torch.zeros([len(h), 0], device='cuda')
        for conv in self.conv_o:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)

        sim_h = sim_graph.ndata['x']
        sim_h = self.linear1_s(sim_h)
        sim_h = self.act(sim_h)
        # sim_h = self.linear2_s(sim_h)
        # sim_h = self.act(sim_h)
        sim_h_final = torch.zeros([len(sim_h), 0], device='cuda')
        for conv in self.conv_s:
            h0 = conv(sim_graph, sim_h)
            sim_h_final = torch.cat([sim_h_final, h0], -1)
        sim_h_final = self.dropout(sim_h_final)

        h_all = torch.cat([h_final, sim_h_final], -1)
        logits = self.mlp(h_all)

        return h_all, logits


class MyDetector(object):
    def __init__(self, train_config, model_config, data):
        feat = data.ndata['x']
        knn_graph = KNNGraph(model_config['k'])
        knn_g = knn_graph(feat, algorithm='bruteforce-sharemem', dist=model_config['dist'])
        knn_g.ndata['x'] = feat
        self.sim_graph = knn_g

        self.train_config = train_config
        self.model_config = model_config
        self.in_feats = data.ndata['x'].shape[1]
        self.train_mask = data.ndata['train_mask'].bool()
        self.val_mask = data.ndata['val_mask'].bool()
        self.test_mask = data.ndata['test_mask'].bool()
        self.train_y = data.ndata['y'][self.train_mask]
        self.val_y = data.ndata['y'][self.val_mask]
        self.test_y = data.ndata['y'][self.test_mask]
        self.source_graph = data
        self.weight = (1 - self.train_y).sum().item() / self.train_y.sum().item()  # sum(0)/ sum(1)
        self.best_val_score = 0
        self.patience_knt = 0
        self.gnn_model = MyModel(in_feats=self.in_feats,
                                 hid_feats=model_config['hid_feats'],
                                 out_feats=2,
                                 num_layers=model_config['num_layers'],
                                 dropout_rate=model_config['dropout_rate'],
                                 activation=model_config['activation']
                                 ).to('cuda')
        eval_metric = roc_auc_score if train_config['metric'] == 'auc' else average_precision_score
        self.tree_model = XGBClassifier(device='cuda',
                                        tree_method='hist',
                                        eval_metric=eval_metric,
                                        n_estimators=model_config['n_estimators'],
                                        eta=model_config['eta'],
                                        subsample=model_config['subsample'],
                                        colsample_bytree=model_config['colsample_bytree'],
                                        reg_lambda=model_config['reg_lambda'],
                                        scale_pos_weight = self.weight
                                        )

    def pretrain(self):
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=self.model_config['learning_rate'])
        for e in range(train_config['epochs']):
            self.gnn_model.train()
            h, logits = self.gnn_model(self.source_graph, self.sim_graph)
            loss = F.cross_entropy(logits[self.train_mask], self.train_y,
                                   weight=torch.tensor([1., self.weight], device='cuda'))

            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.model_config['dropout_rate'] > 0:
                self.gnn_model.eval()
                with torch.no_grad():
                    h, logits = self.gnn_model(self.source_graph, self.sim_graph)

            probs = logits.softmax(1)[:, 1]
            val_y_prob = probs[self.val_mask]
            val_y_pred = prob_convert_pred(val_y_prob.cpu().detach().numpy(), thres=0.5)
            val_score = eval_metric(self.val_y.cpu().detach().numpy(), val_y_prob.cpu().detach().numpy(), val_y_pred)

            if val_score[train_config['metric']] > self.best_val_score:
                self.patience_knt = 0
                self.best_val_score = val_score[train_config['metric']]
                self.best_model = copy.deepcopy(self.gnn_model)
                self.best_emb = h.detach()
            else:
                self.patience_knt += 1
                if self.patience_knt > train_config['patience']:
                    break
        return self.best_val_score

    def train(self):
        torch.cuda.empty_cache()
        all_emb = torch.cat((self.source_graph.ndata['x'], self.best_emb), dim=1)
        self.train_emb = all_emb[self.train_mask]
        self.val_emb = all_emb[self.val_mask]
        self.test_emb = all_emb[self.test_mask]
        self.tree_model.fit(self.train_emb, self.train_y)
        val_y_prob = self.tree_model.predict_proba(self.val_emb)[:, 1]
        val_y_pred = prob_convert_pred(val_y_prob, thres=0.5)
        val_score = eval_metric(self.val_y.cpu().numpy(), val_y_prob, val_y_pred)
        self.best_pr_thres = val_score['best_pr_thres']
        return val_score[train_config['metric']]

    def test(self):
        test_y_prob = self.tree_model.predict_proba(self.test_emb)[:, 1]
        test_y_pred = prob_convert_pred(test_y_prob, thres=self.best_pr_thres)
        test_score = eval_metric(self.test_y.cpu().numpy(), test_y_prob, test_y_pred)
        return test_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dgraphfin', help='fdcompcn|ffsd|elliptic|dgraphfin')
    parser.add_argument('--model', type=str, default='model_xgb_pp')
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    train_config = {
        'epochs': 200,
        'patience': 50,
        'metric': 'ap'
    }
    param_space = {
        'num_layers': [1, 2],
        'hid_feats': [16, 32, 64],
        'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
        'dropout_rate': [0, 0.1, 0.2, 0.3],
        'learning_rate': 10 ** np.linspace(-3, -1, 1000),
        'k': range(10, 101),
        'dist': ['euclidean', 'cosine'],
        'n_estimators': list(range(10, 301)),
        'eta': 0.5 * 10 ** np.linspace(-1, 0, 100),
        'reg_lambda': [0, 1, 10],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0]
    }

    start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_path = r'../../checkpoints/{}/{}/{}/{}'.format(args.dataset, args.model, train_config['metric'],
                                                              start_wall_time)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    model1_path = os.path.join(checkpoint_path, 'model1.pkl')
    model2_path = os.path.join(checkpoint_path, 'model2.pkl')
    config_path = os.path.join(checkpoint_path, 'config.yml')
    result_path = os.path.join(checkpoint_path, 'results.xlsx')

    time_cost = 0
    best_val_score = 0
    data = load_data(args.dataset)
    data = dgl.to_bidirected(data, copy_ndata=True)
    data = dgl.add_self_loop(data).to('cuda')

    print("----- Dataset: {}, Model: {}, Metric: {} -----".format(args.dataset, args.model, train_config['metric']))
    for t in range(args.trials):
        start_time = time.time()
        model_config = {}
        for k, v in param_space.items():
            model_config[k] = random.choice(v)
        detector = MyDetector(train_config, model_config, data)
        pre_val_score = detector.pretrain()
        val_score = detector.train()
        end_time = time.time()
        time_cost = end_time - start_time

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(detector.best_model.state_dict(), model1_path)
            joblib.dump(detector.tree_model, model2_path)
            test_score = detector.test()

            with open(config_path, 'w') as f:
                yaml.dump(model_config, f)
            df_result = pd.DataFrame(test_score, index=[0])
            df_result['time_cost'] = time_cost
            df_result['dataset'] = args.dataset
            df_result['model'] = args.model
            df_result.to_excel(result_path)

        print("Trial {}, Val Score: {:.4f}, Time Cost: {:.3f}".format(t, val_score, time_cost))
    print('end')
