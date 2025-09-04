import os
import shutil
import torch

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from graphgen_utils import get_model_attribute, save_graphs

LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_path = 'model_save/' + 'DFScodeRNN_MIMIC-Breast_2025-04-18 16:55:56/DFScodeRNN_MIMIC-Breast_3940.dat'    # 'model_name'

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.count = 40000    # 2560   # Number of graphs to generate
        self.batch_size = 25    # 32  # Must be a factor of count

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


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    if eval_args.generate_graphs:
        generate_graphs(eval_args)
        print('Graphs generated')
