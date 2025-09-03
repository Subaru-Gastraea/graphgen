import os
import pickle
import networkx as nx
from tqdm import tqdm
import argparse
import pathlib
import sys

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/graphgen/datasets/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))     # ddx-on-ehr/models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))  # ddx-on-ehr/

from utils.graph_proc import GraphDataset

def write_graph_txt(graph, subGraphID, txt_path):
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    
    with open(txt_path, 'a') as f:
        f.write(f"#{subGraphID}\n")
        f.write(f"{node_count}\n")
        for n in range(node_count):
            node_label = graph.nodes[n].get('type', None)
            f.write(f"{node_label}\n")
        f.write(f"{edge_count}\n")
        for (u, v, data) in graph.edges(data=True):
            edge_label = data.get('edge_type', None)
            f.write(f"{u} {v} {edge_label}\n")
        f.write("\n")

def gen_subgraphs(args):
    target_label = args.target_label
    graph_cnt = 0
    filter_graph_cnt = 0

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    print(f'Generating input txt for label {target_label}')

    if os.path.exists(sample_dataset_path):
         with open(sample_dataset_path, 'rb') as f:
            sample_dataset = pickle.load(f)
   
    dataset = GraphDataset(sample_dataset, dev=False, project_root=project_root)
    del sample_dataset
    dataset.set_split('train')

    # Create output folder
    output_folder = pathlib.Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    txt_path = output_folder / "all_graphs.txt"

    for i, graph_info in enumerate(tqdm(dataset, total=len(dataset), desc="Generating subgraphs")):
        graph, label = graph_info
        subGraphID = i

        if label == target_label:
            graph_cnt += 1
            # Filter out graphs with more than 1000 nodes
            if len(graph.nodes) > 1000:
                filter_graph_cnt += 1
                continue

            mapping = {node: j for j, node in enumerate(graph.nodes)}
            graph = nx.relabel_nodes(graph, mapping)
            write_graph_txt(graph, subGraphID, txt_path)

            if args.devm and graph_cnt == 100: # Test with 100 graphs
                break

    print('Total graphs:', graph_cnt)
    print('Filtered graphs:', filter_graph_cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make graphs into raw format for process_dataset.py")
    parser.add_argument('--target_label', type=int, default=0, help='Generate .txt only for this label')
    parser.add_argument('--output', type=str, default='MIMIC-Breast/', required=True)
    parser.add_argument('--devm', action='store_true', default=False, help='develop mode [add "--devm" to enable]')
    args = parser.parse_args()

    gen_subgraphs(args)