import os
import random
import shutil
from statistics import mean
import torch

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from graphgen_utils import get_model_attribute, load_graphs, save_graphs

import metrics.stats
import sys
import pickle
import pathlib

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/graphgen/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # ddx-on-ehr/models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # ddx-on-ehr/

from utils.graph_proc import GraphDataset

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import lil_matrix
from tqdm import tqdm

LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model_path = 'model_save/' + 'DFScodeRNN_MIMIC-Breast_2025-04-14 15:05:33/DFScodeRNN_MIMIC-Breast_4200.dat' # 'model_name'

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = False

        self.count = 100    # 2560   # Number of graphs to generate
        self.batch_size = 32  # Must be a factor of count

        self.metric_eval_batch_size = 256

        # Specific DFScodeRNN
        self.max_num_edges = 50

        # Specific to GraphRNN
        self.min_num_node = 0
        self.max_num_node = 40

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
            self.train_args.time + '/' + str(self.num_epochs) + '/'


def patch_graph(graph):
    for u in graph.nodes():
        graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]

    return graph


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Edge label - {:.6f}'.format(
        mean(node_label_mmd), mean(edge_label_mmd)
    ))
    print('MMD Joint Node label and degree - {:.6f}'.format(
        mean(node_label_and_degree)
    ))
    print(LINE_BREAK)


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    if eval_args.generate_graphs:
        generate_graphs(eval_args)
        print('Graphs generated')

    # Load generated graphs
    graphs_pred_indices = []
    if not eval_args.generate_graphs:
        for name in os.listdir(eval_args.current_graphs_save_path):
            if name.endswith('.dat'):
                graphs_pred_indices.append(len(graphs_pred_indices))
    else:
        graphs_pred_indices = [i for i in range(eval_args.count)]

    graphs_gen_label0 = load_graphs(
            eval_args.current_graphs_save_path, graphs_pred_indices)
    
    # Load test graphs
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    if os.path.exists(sample_dataset_path):
         with open(sample_dataset_path, 'rb') as f:
            sample_dataset = pickle.load(f)
   
    dataset = GraphDataset(sample_dataset, dev=False, project_root=project_root)
    del sample_dataset
    dataset.set_split('test')

    # 1. 將graph轉為adjacency matrix，計算相似度 (同位置match的1個數/總個數)
    def graph_similarity(G1, G2):
        all_nodes = set(G1.nodes()) | set(G2.nodes())
        node_index = {node: i for i, node in enumerate(all_nodes)}
        size = len(all_nodes)

        def build_adj_matrix(G):
            # Use a sparse matrix to save memory
            mat = lil_matrix((size, size), dtype=np.int8)
            for u, v in G.edges():
                i, j = node_index[u], node_index[v]
                mat[i, j] = mat[j, i] = 1
            return mat.toarray()

        adj1 = build_adj_matrix(G1)
        adj2 = build_adj_matrix(G2)
        match_count = np.sum((adj1 == 1) & (adj2 == 1))
        total_count = size * size
        return match_count / total_count

    # 2. 每個testing graph和label=0的100張生成圖，計算平均相似度
    pred_labels = []
    sim_scores = []
    test_labels = []
    for test_g, test_label in tqdm(dataset, desc="Evaluating test graphs"):
        test_labels.append(test_label if test_label == 0 else 1)  # label=0以外的，合併成label=1

        sims = [graph_similarity(test_g, gen_g) for gen_g in graphs_gen_label0]
        avg_sim = np.mean(sims)
        sim_scores.append(avg_sim)
        # 3. 平均相似度>=0.5的testing graph，判斷為label=0
        pred_labels.append(0 if avg_sim >= 0.5 else 1)

    # 4. 用合適的metrics，呈現testing效果
    acc = accuracy_score(test_labels, pred_labels)
    prec = precision_score(test_labels, pred_labels, pos_label=0)
    rec = recall_score(test_labels, pred_labels, pos_label=0)
    f1 = f1_score(test_labels, pred_labels, pos_label=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (label=0): {prec:.4f}")
    print(f"Recall (label=0): {rec:.4f}")
    print(f"F1-score (label=0): {f1:.4f}")

    # random.seed(123)

    # graphs = []
    # for name in os.listdir(train_args.current_dataset_path):
    #     if name.endswith('.dat'):
    #         graphs.append(len(graphs))

    # random.shuffle(graphs)
    # graphs_test_indices = graphs[int(0.90 * len(graphs)):]
    # graphs_train_indices = graphs[:int(0.90 * len(graphs))]

    # graphs_pred_indices = []
    # if not eval_args.generate_graphs:
    #     for name in os.listdir(eval_args.current_graphs_save_path):
    #         if name.endswith('.dat'):
    #             graphs_pred_indices.append(len(graphs_pred_indices))
    # else:
    #     graphs_pred_indices = [i for i in range(eval_args.count)]

    # print('Evaluating {}, run at {}, epoch {}'.format(
    #     train_args.fname, train_args.time, eval_args.num_epochs))

    # print('Graphs generated - {}'.format(len(graphs_pred_indices)))

    # metrics.stats.novelity(
    #     train_args.current_dataset_path, graphs_train_indices, eval_args.current_graphs_save_path,
    #     graphs_pred_indices, train_args.temp_path, timeout=60)

    # metrics.stats.uniqueness(
    #     eval_args.current_graphs_save_path,
    #     graphs_pred_indices, train_args.temp_path, timeout=120)

    # # exit()

    # node_count_avg_ref, node_count_avg_pred = [], []
    # edge_count_avg_ref, edge_count_avg_pred = [], []

    # degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
    # node_label_mmd, edge_label_mmd, node_label_and_degree = [], [], []

    # print(len(graphs_test_indices))
    # for i in range(0, len(graphs_pred_indices), eval_args.metric_eval_batch_size):
    #     batch_size = min(eval_args.metric_eval_batch_size,
    #                      len(graphs_pred_indices) - i)

    #     graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
    #     graphs_ref = load_graphs(
    #         train_args.current_dataset_path, graphs_ref_indices)

    #     graphs_ref = [patch_graph(g) for g in graphs_ref]

    #     graphs_pred = load_graphs(
    #         eval_args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])

    #     graphs_pred = [patch_graph(g) for g in graphs_pred]

    #     node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
    #     node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

    #     edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
    #     edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

    #     degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
    #     clustering_mmd.append(
    #         metrics.stats.clustering_stats(graphs_ref, graphs_pred))
    #     orbit_mmd.append(metrics.stats.orbit_stats_all(
    #         graphs_ref, graphs_pred))

    #     nspdk_mmd.append(metrics.stats.nspdk_stats(graphs_ref, graphs_pred))

    #     node_label_mmd.append(
    #         metrics.stats.node_label_stats(graphs_ref, graphs_pred))
    #     edge_label_mmd.append(
    #         metrics.stats.edge_label_stats(graphs_ref, graphs_pred))
    #     node_label_and_degree.append(
    #         metrics.stats.node_label_and_degree_joint_stats(graphs_ref, graphs_pred))

    #     print('Running average of metrics:\n')

    #     print_stats(
    #         node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
    #         degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
    #         edge_label_mmd, node_label_and_degree
    #     )

    # print('Evaluating {}, run at {}, epoch {}'.format(
    #     train_args.fname, train_args.time, eval_args.num_epochs))

    # print_stats(
    #     node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
    #     degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
    #     edge_label_mmd, node_label_and_degree
    # )
