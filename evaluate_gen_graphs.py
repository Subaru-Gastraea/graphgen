import os
import torch

from graphgen_utils import get_model_attribute, load_graphs

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


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model_path = 'model_save/' + 'DFScodeRNN_MIMIC-Breast_2025-04-18 16:55:56/DFScodeRNN_MIMIC-Breast_3940.dat' # 'model_name'

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

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


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    # Load generated graphs
    graphs_pred_indices = []
    for name in os.listdir(eval_args.current_graphs_save_path):
        if name.endswith('.dat'):
            graphs_pred_indices.append(len(graphs_pred_indices))

    graphs_gen_label0 = load_graphs(
            eval_args.current_graphs_save_path, graphs_pred_indices)
    
    print("Rename node labels to 'type' in generated graphs")
    for gen_g in graphs_gen_label0:
        for node in gen_g.nodes:
            if 'label' in gen_g.nodes[node]:
                gen_g.nodes[node]['type'] = gen_g.nodes[node].pop('label')

    # Load test graphs
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    if os.path.exists(sample_dataset_path):
         with open(sample_dataset_path, 'rb') as f:
            sample_dataset = pickle.load(f)
   
    dataset = GraphDataset(sample_dataset, dev=False, project_root=project_root)
    del sample_dataset
    dataset.set_split('test')

    # 每個testing graph和label=0的100張生成圖，計算平均相似度
    pred_labels = []
    sim_scores = []
    test_labels = []
    for test_g, test_label in tqdm(dataset, desc="Evaluating test graphs"):
        test_labels.append(test_label if test_label == 0 else 1)  # label=0以外的，合併成label=1

        # [0]: lab_similarity
        sims = [dataset.calculate_similarity(test_g, gen_g, node_attr_name='type')[0] for gen_g in graphs_gen_label0]
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
