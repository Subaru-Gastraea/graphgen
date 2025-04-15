import os
import pickle
import networkx as nx
from tqdm import tqdm
import argparse
import pathlib

def write_graph_txt(graph, subGraphID, output_folder):
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    txt_path = output_folder / "all_graphs.txt"
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
    graph_dataset_path = args.dataset
    target_label = args.target_label
    graph_cnt = 0   # for develop mode

    graphs_path = os.listdir(graph_dataset_path)

    def get_graph_info():
        for graph_path in graphs_path:
            path = os.path.join(graph_dataset_path, graph_path)
            with open(path, 'rb') as f:
                graph_sample = pickle.load(f)
                yield graph_sample['graph'], graph_sample['labels']

    for i, graph_info in enumerate(tqdm(get_graph_info(), total=len(graphs_path), desc="Generating subgraphs")):
        graph, label = graph_info
        subGraphID = i
        label_N = label.index(1)

        # Filter out graphs with more than 1000 nodes
        if len(graph.nodes) > 1000:
            continue

        if label_N == target_label:
            mapping = {node: j for j, node in enumerate(graph.nodes)}
            graph = nx.relabel_nodes(graph, mapping)
            write_graph_txt(graph, subGraphID, args.output)

            if args.devm:
                graph_cnt += 1
                if graph_cnt == 100: # Test with 100 graphs
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make graphs into raw format for process_dataset.py")
    parser.add_argument('--dataset', type=str, default='patients_graphs/', required=True)
    parser.add_argument('--target_label', type=int, default=0, help='Generate .txt only for this label')
    parser.add_argument('--output', type=str, default='MIMIC-Breast/', required=True)
    parser.add_argument('--devm', action='store_true', default=False, help='develop mode [add "--devm" to enable]')
    args = parser.parse_args()

    gen_subgraphs(args)