import os
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset


class FdcompcnData(DGLDataset):
    def __init__(self, raw_dir=None, save_dir=None):
        super().__init__(name='fdcompcn', raw_dir=raw_dir, save_dir=save_dir)

    def process(self):
        graph = dgl.load_graphs(os.path.join(self.raw_dir, 'comp.dgl'))[0][0]
        graph = dgl.edge_type_subgraph(graph, ['homo'])
        labels = graph.ndata['label']

        n = graph.num_nodes()
        index = list(range(n))
        train_mask = torch.zeros(n).bool()
        val_mask = torch.zeros(n).bool()
        test_mask = torch.zeros(n).bool()
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=0.7, random_state=42, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                train_size=0.5, random_state=42, shuffle=True)
        train_mask[idx_train] = True
        val_mask[idx_valid] = True
        test_mask[idx_test] = True

        # Add information to the graph
        graph.ndata['y'] = labels.to(dtype=torch.int64)
        graph.ndata['x'] = graph.ndata['feature'].to(dtype=torch.float32)
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        graph.ndata['mask'] = train_mask + val_mask + test_mask
        graph.ndata['node_id'] = torch.arange(n)

        dgl.save_graphs(os.path.join(self.save_dir, 'fdcompcn.dgl'), [graph])

class FFSDData(DGLDataset):
    def __init__(self, raw_dir=None, save_dir=None):
        super().__init__(name='ffsd', raw_dir=raw_dir, save_dir=save_dir)

    def process(self):
        graph = dgl.load_graphs(os.path.join(self.raw_dir, 'FFSD.bin'))[0][0]
        x = graph.ndata['feat']
        y = graph.ndata['label']

        # Divide the training set, validation set, and test set
        idx_labeled = y != 2
        n = len(y)
        index = list(range(n))
        train_mask = torch.zeros(n).bool()
        val_mask = torch.zeros(n).bool()
        test_mask = torch.zeros(n).bool()
        idx_train, idx_rest, y_train, y_rest = train_test_split(np.array(index)[idx_labeled], y[idx_labeled],
                                                                stratify=y[idx_labeled],train_size=0.7,
                                                                random_state=42, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                train_size=0.5, random_state=42,
                                                                shuffle=True)
        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1

        # Build graph
        graph = dgl.graph((graph.edges()[0], graph.edges()[1]))

        # Add information to the graph
        graph.ndata['y'] = y.to(dtype=torch.int64)
        graph.ndata['x'] = x.to(dtype=torch.float32)
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        graph.ndata['mask'] = train_mask + val_mask + test_mask
        graph.ndata['node_id'] = torch.arange(n)

        dgl.save_graphs(os.path.join(self.save_dir, 'ffsd.dgl'), [graph])

        return graph


class EllipticData(DGLDataset):
    def __init__(self, raw_dir=None, save_dir=None):
        super().__init__(name='elliptic', raw_dir=raw_dir, save_dir=save_dir)

    def process(self):
        feat_df = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_features.csv'), header=None)
        edge_df = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv'))
        class_df = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_classes.csv'))

        columns = {0: 'txId', 1: 'time_step'}
        feat_df = feat_df.rename(columns=columns)
        # feat_df['time_step'] = feat_df['time_step'] - 1
        x = torch.from_numpy(feat_df.loc[:, 2:].values).to(torch.float)

        # There exists 3 different classes in the dataset: 0=licit,  1=illicit, 2=unknown
        mapping = {'unknown': 2, '1': 1, '2': 0}
        class_df['class'] = class_df['class'].map(mapping)
        y = torch.from_numpy(class_df['class'].values)

        mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
        txId2t = feat_df.set_index('txId')['time_step']
        edge_df['time_step'] = edge_df['txId1'].map(txId2t)
        edge_df['txId1'] = edge_df['txId1'].map(mapping)
        edge_df['txId2'] = edge_df['txId2'].map(mapping)
        edge_index = torch.from_numpy(edge_df[['txId1', 'txId2']].values).t().contiguous()
        edge_time = torch.from_numpy(edge_df['time_step'].values)
        edge_id = torch.arange(len(edge_time))
        time_step = torch.from_numpy(feat_df['time_step'].values)

        # 按时间戳划分（原文划分）
        train_mask = (time_step < 24) & (y != 2)
        val_mask = (time_step >= 24) & (time_step < 34) & (y != 2)
        test_mask = (time_step >= 34) & (y != 2)

        graph = dgl.graph((edge_index[0, :], edge_index[1, :]))

        graph.ndata['x'] = x
        graph.ndata['y'] = y
        graph.ndata['node_time'] = time_step
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        graph.ndata['mask'] = train_mask + val_mask + test_mask
        graph.ndata['node_id'] = torch.arange(len(y))

        graph.edata['edge_index'] = edge_index.T
        graph.edata['edge_time'] = edge_time
        graph.edata['edge_id'] = edge_id

        graph = dgl.node_subgraph(graph, graph.ndata['mask'])
        dgl.save_graphs(os.path.join(self.save_dir, 'elliptic.dgl'), [graph])


class DGraphfinData(DGLDataset):

    def __init__(self, raw_dir=None, save_dir=None):
        super().__init__(name='dgraphfin', raw_dir=raw_dir, save_dir=save_dir)

    def process(self):
        loader = np.load(os.path.join(self.raw_dir, 'dgraphfin.npz'))
        x = torch.from_numpy(loader['x']).to(torch.float)
        y = torch.from_numpy(loader['y']).to(torch.long)
        y[y > 1] = 2

        edge_index = torch.from_numpy(loader['edge_index']).to(torch.long)
        edge_type = torch.from_numpy(loader['edge_type']).to(torch.long)
        edge_time = torch.from_numpy(loader['edge_timestamp']).to(torch.long)
        edge_id = torch.arange(len(edge_time))

        # # 原文划分
        idx_train = torch.from_numpy(loader['train_mask']).to(torch.long)
        idx_val = torch.from_numpy(loader['valid_mask']).to(torch.long)
        idx_test = torch.from_numpy(loader['test_mask']).to(torch.long)
        train_mask = torch.zeros(len(y)).bool()
        val_mask = torch.zeros(len(y)).bool()
        test_mask = torch.zeros(len(y)).bool()
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        graph = dgl.graph((edge_index[:, 0], edge_index[:, 1]))
        graph.ndata['x'] = x
        graph.ndata['y'] = y
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        graph.ndata['mask'] = train_mask + val_mask + test_mask
        graph.ndata['node_id'] = torch.arange(len(y))

        graph.edata['edge_index'] = edge_index
        graph.edata['edge_type'] = edge_type
        graph.edata['edge_time'] = edge_time
        graph.edata['edge_id'] = edge_id

        graph = dgl.node_subgraph(graph, graph.ndata['mask'])
        dgl.save_graphs(os.path.join(self.save_dir, 'dgraphfin.dgl'), [graph])


if __name__ == '__main__':
    # FdcompcnData(raw_dir=r'datasets/fdcompcn/raw', save_dir=r'datasets/fdcompcn/processed')
    fdcompcn = dgl.load_graphs(r'./datasets/fdcompcn/processed/fdcompcn.dgl')[0][0]
    print(fdcompcn)

    # FFSDData(raw_dir=r'datasets/ffsd/raw', save_dir=r'datasets/ffsd/processed')
    ffsd = dgl.load_graphs(r'./datasets/ffsd/processed/ffsd.dgl')[0][0]
    print(ffsd)

    # EllipticData(raw_dir=r'datasets/elliptic/raw', save_dir=r'datasets/elliptic/processed')
    elliptic = dgl.load_graphs(r'./datasets/elliptic/processed/elliptic.dgl')[0][0]
    print(elliptic)
    #
    # DGraphfinData(raw_dir=r'datasets/dgraphfin/raw', save_dir=r'datasets/dgraphfin/processed')
    dgraphfin = dgl.load_graphs(r'./datasets/dgraphfin/processed/dgraphfin.dgl')[0][0]
    print(dgraphfin)

    print('end')
