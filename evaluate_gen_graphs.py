import os
import torch

from graphgen_utils import get_model_attribute, load_graphs

import sys
import pickle
import pathlib

import pandas as pd

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/graphgen/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # ddx-on-ehr/models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # ddx-on-ehr/

from utils.graph_proc import GraphDataset

import numpy as np
from tqdm import tqdm

import argparse
from sklearn.metrics import classification_report, confusion_matrix

import time

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
    parser.add_argument('--train_or_test', type=str, default='test', help='Compare with train or test graphs')
    parser.add_argument('--time_slice_pct', type=float, default=1.0, help='Percentage of time for slicing test graphs')
    args = parser.parse_args()

    model_paths = {
        0: 'model_save/' + 'DFScodeRNN_MIMIC-Breast_2025-04-18 16:55:56/DFScodeRNN_MIMIC-Breast_3940.dat',
        1: 'model_save/' + 'DFScodeRNN_MIMIC-Lung_2025-04-18 01:22:37/DFScodeRNN_MIMIC-Lung_3200.dat',
        2: 'model_save/' + 'DFScodeRNN_MIMIC-Ovary_2025-04-18 01:28:11/DFScodeRNN_MIMIC-Ovary_4000.dat',
        3: 'model_save/' + 'DFScodeRNN_MIMIC-Colon_2025-04-18 01:30:38/DFScodeRNN_MIMIC-Colon_4000.dat',
        4: 'model_save/' + 'DFScodeRNN_MIMIC-Prostate_2025-04-18 01:32:00/DFScodeRNN_MIMIC-Prostate_3940.dat'
    }

    # Load test graphs
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    if os.path.exists(sample_dataset_path):
        with open(sample_dataset_path, 'rb') as f:
            sample_dataset = pickle.load(f)

    print(f"Loading {args.train_or_test} graphs...")
    dataset = GraphDataset(sample_dataset, dev=False, project_root=project_root)
    del sample_dataset
    dataset.set_split(args.train_or_test)
    
    if args.train_or_test == 'test':
        print("Time slice percentage:", args.time_slice_pct)
        print("Slicing graphs...")
        dataset.time_slice_graphs(args.time_slice_pct)
    
    duration = 0.0
    start_time = time.time()

    # Load generated graphs
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
        
        print("Rename node labels to 'type' in generated graphs")
        for gen_g in gen_graphs:
            for node in gen_g.nodes:
                if 'label' in gen_g.nodes[node]:
                    gen_g.nodes[node]['type'] = gen_g.nodes[node].pop('label')

        print("Rename edge labels to 'edge_type' in generated graphs")
        for gen_g in gen_graphs:
            for edge in gen_g.edges:
                if 'label' in gen_g.edges[edge]:
                    gen_g.edges[edge]['edge_type'] = gen_g.edges[edge].pop('label')

        label_gen_graphs[label] = gen_graphs

    end_time = time.time()
    duration += end_time - start_time

    FILT_SIM_ZERO = True  # 是否過濾相似度為0的圖
    print(f"Filtering zero similarity: {FILT_SIM_ZERO}")

    sim_path_root = pathlib.Path(f'non_time_aware_sims/')
    if args.train_or_test == 'train':
        sim_path = sim_path_root / 'train_graph_sims/'
    else:
        sim_path = sim_path_root / f'test_graph_sims/time_slice_pct_{args.time_slice_pct}/'
    
    sim_path.mkdir(parents=True, exist_ok=True)

    num_labels = len(label_gen_graphs)  # 5 , exclude label=5 (others)
    num_gen_graphs = min([len(gen_graphs) for gen_graphs in label_gen_graphs.values()])  # 每個label有100張生成圖
    pred_labels = []
    true_labels = []
    all_norm_avg_sims = []

    print(f"Number of generated graphs per label: {num_gen_graphs}")

    for idx, (g, label) in enumerate(tqdm(dataset, desc=f"Calculating {args.train_or_test} graph similarities")):
        true_labels.append(label)
        graph_path = sim_path / f"{args.train_or_test}_graph_{idx}.npy"

        # shape: (num_labels, num_gen_graphs)
        if os.path.exists(graph_path):
            sim_matrix = np.load(graph_path)
        else:
            start_time = time.time()

            sim_matrix = np.zeros((num_labels, num_gen_graphs))
            calc_sim = dataset.calculate_similarity
            # Use numpy vectorization where possible, and avoid repeated attribute lookups
            for lbl in range(num_labels):
                gen_graphs = label_gen_graphs[lbl][:num_gen_graphs]
                # Use list comprehension, but pre-bind the function for less overhead
                sims = [calc_sim(g, gen_g, node_attr_name='type') for gen_g in gen_graphs]
                sim_matrix[lbl, :] = sims

            end_time = time.time()
            duration += end_time - start_time

            np.save(graph_path, sim_matrix)

        start_time = time.time()

        if FILT_SIM_ZERO:
            # 過濾每個label下相似度為0的生成圖
            filtered_sim_matrix = []
            for lbl in range(num_labels):
                nonzero = sim_matrix[lbl] > 0
                if np.any(nonzero):
                    filtered_sim_matrix.append(sim_matrix[lbl][nonzero])
                else:
                    filtered_sim_matrix.append(np.array([0]))
        else:
            filtered_sim_matrix = [sim_matrix[lbl] for lbl in range(num_labels)]

        # 計算每個label的平均相似度
        avg_sims = np.array([np.mean(filtered_sim_matrix[lbl]) for lbl in range(num_labels)])

        # softmax normalization
        exp_sims = np.exp(avg_sims)
        norm_avg_sims = exp_sims / np.sum(exp_sims)

        # Append a 0 for label 5 (others)
        fix_norm_avg_sims = np.append(norm_avg_sims, 0.0)

        # Add bias to label 5 (others) (_v2)
        avg_val = np.mean(norm_avg_sims)
        bias = 0.05
        if np.max(norm_avg_sims) <= avg_val:    # label 0 ~ 4 are 0.2
            fix_norm_avg_sims[-1] = min(avg_val + bias, 0.25)  # 不超過 0.25

        # Collect all norm_avg_sims for saving later
        all_norm_avg_sims.append(fix_norm_avg_sims)

        end_time = time.time()
        duration += end_time - start_time

        # multi-class prediction
        # pred_labels.append(int(np.argmax(fix_norm_avg_sims)))

    event = "Calculate similarity for test graphs"
    log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event}: {duration:.6f} seconds\n"
    with open('logfile.txt', 'a') as f:
        f.write(log_message)

    # Save all norm_avg_sims to a CSV file
    all_norm_avg_sims_df = pd.DataFrame(all_norm_avg_sims)
    all_norm_avg_sims_df.to_csv(sim_path / f'{args.train_or_test}_norm_avg_sims.csv', index=False, header=False)

    # if args.train_or_test == 'test':
    #     # 評估: multi-class classification
    #     print(classification_report(true_labels, pred_labels, labels=list(range(num_labels+1)), digits=4, zero_division=0))
    #     print("Confusion matrix:")
    #     print(confusion_matrix(true_labels, pred_labels, labels=list(range(num_labels+1))))

    #     # Save classification report to file
    #     report = classification_report(true_labels, pred_labels, labels=list(range(num_labels+1)), digits=4, output_dict=False, zero_division=0)
    #     with open('classification_report.txt', 'w') as f:
    #         f.write(report)

    #     # Save confusion matrix to file
    #     cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_labels+1)))
    #     cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(num_labels+1)],
    #                             columns=[f"Pred_{i}" for i in range(num_labels+1)])
    #     cm_df.to_csv('confusion_matrix.csv')
