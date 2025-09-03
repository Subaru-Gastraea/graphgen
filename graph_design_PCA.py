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

import argparse
import random

random.seed(42)  # For reproducibility

class ArgsEvaluate():
    def __init__(self, model_path):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_path = model_path

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
            self.train_args.time + '/' + str(self.num_epochs) + '/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_slice_pct', type=float, default=1.0, help='Percentage of time for slicing test graphs')
    args = parser.parse_args()

    model_paths = {
        0: 'model_save/' + 'DFScodeRNN_MIMIC-Breast_2025-04-18 16:55:56/DFScodeRNN_MIMIC-Breast_3940.dat',
        1: 'model_save/' + 'DFScodeRNN_MIMIC-Lung_2025-04-18 01:22:37/DFScodeRNN_MIMIC-Lung_3200.dat',
        2: 'model_save/' + 'DFScodeRNN_MIMIC-Ovary_2025-04-18 01:28:11/DFScodeRNN_MIMIC-Ovary_4000.dat',
        3: 'model_save/' + 'DFScodeRNN_MIMIC-Colon_2025-04-18 01:30:38/DFScodeRNN_MIMIC-Colon_4000.dat',
        4: 'model_save/' + 'DFScodeRNN_MIMIC-Prostate_2025-04-18 01:32:00/DFScodeRNN_MIMIC-Prostate_3940.dat'
    }

    # Load graphs
    print("Loading original graphs...")
    label_graphs = {}
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    if os.path.exists(sample_dataset_path):
        with open(sample_dataset_path, 'rb') as f:
            sample_dataset = pickle.load(f)

    dataset = GraphDataset(sample_dataset, dev=False, split=False, project_root=project_root)
    del sample_dataset
    
    if args.time_slice_pct < 1.0:
        print("Time slice percentage:", args.time_slice_pct)
        print("Slicing graphs...")
        dataset.time_slice_graphs(args.time_slice_pct)

    for sample in dataset.graph_samples:
        label = sample['labels'].index(1)
        if label == 5:  # Skip 'Other' label
            continue

        graph = sample['graph']
        if label not in label_graphs:
            label_graphs[label] = []
        label_graphs[label].append(graph)

    # Randomly sample a subset of graphs for each label
    sample_num = 100
    for label, graphs in label_graphs.items():
        random.shuffle(graphs)
        label_graphs[label] = graphs[:sample_num]

    print(f"Number of labels: {len(label_graphs)}")
    print(f"Number of graphs loaded: {sum(len(g) for g in label_graphs.values())}")

    dataset.plot_graphs_PCA(label_graphs, node_attr_name='type', save_root='./', fname='orig_graph_pca')

    # Load generated graphs
    print("Loading generated graphs...")
    label_gen_graphs = {}
    for label, model_path in model_paths.items():
        eval_args = ArgsEvaluate(model_path=model_path)
        train_args = eval_args.train_args

        print('Loading generated graphs from {}, run at {}, epoch {}'.format(
            train_args.fname, train_args.time, eval_args.num_epochs))

        graphs_pred_indices = []
        for name in os.listdir(eval_args.current_graphs_save_path):
            if name.endswith('.dat'):
                graphs_pred_indices.append(len(graphs_pred_indices))

        gen_graphs = load_graphs(
                eval_args.current_graphs_save_path, graphs_pred_indices)

        label_gen_graphs[label] = gen_graphs

    dataset.plot_graphs_PCA(label_gen_graphs, node_attr_name='label', save_root='./', fname='gen_graph_pca')