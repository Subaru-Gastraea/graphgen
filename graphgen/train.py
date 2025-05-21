import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
import networkx as nx

from graphgen.model import create_model
from graphgen_utils import load_model, get_model_attribute
from dfscode.dfs_wrapper import graph_from_dfscode

import numpy as np
from tqdm import tqdm
import pathlib as Pathlib

seed = 25
torch.manual_seed(seed)
np.random.seed(seed)

def evaluate_loss(args, model, data, feature_map):
    x_len_unsorted = data['len'].to(args.device)
    x_len_max = max(x_len_unsorted)
    batch_size = x_len_unsorted.size(0)

    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)

    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    # Prepare targets with end_tokens already there
    t1 = torch.index_select(
        data['t1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    t2 = torch.index_select(
        data['t2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    v1 = torch.index_select(
        data['v1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    e = torch.index_select(
        data['e'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    v2 = torch.index_select(
        data['v2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)

    x_t1, x_t2 = F.one_hot(t1, num_classes=max_nodes +
                           2)[:, :, :-1], F.one_hot(t2, num_classes=max_nodes + 2)[:, :, :-1]
    x_v1, x_v2 = F.one_hot(v1, num_classes=len_node_vec +
                           1)[:, :, :-1], F.one_hot(v2, num_classes=len_node_vec + 1)[:, :, :-1]
    x_e = F.one_hot(e, num_classes=len_edge_vec + 1)[:, :, :-1]

    x_target = torch.cat((x_t1, x_t2, x_v1, x_e, x_v2), dim=2).float()

    # initialize dfs_code_rnn hidden according to batch size
    model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
        batch_size=batch_size)

    # Teacher forcing: Feed the target as the next input
    # Start token is all zeros
    dfscode_rnn_input = torch.cat(
        (torch.zeros(batch_size, 1, feature_len, device=args.device), x_target[:, :-1, :]), dim=1)

    # Forward propogation
    dfscode_rnn_output = model['dfs_code_rnn'](
        dfscode_rnn_input, input_len=x_len.cpu().long() + 1)

    # Evaluating dfscode tuple
    timestamp1 = model['output_timestamp1'](dfscode_rnn_output)
    timestamp2 = model['output_timestamp2'](dfscode_rnn_output)
    vertex1 = model['output_vertex1'](dfscode_rnn_output)
    edge = model['output_edge'](dfscode_rnn_output)
    vertex2 = model['output_vertex2'](dfscode_rnn_output)

    if args.loss_type == 'BCE':
        x_pred = torch.cat(
            (timestamp1, timestamp2, vertex1, edge, vertex2), dim=2)

        # Cleaning the padding i.e setting it to zero
        x_pred = pack_padded_sequence(x_pred, x_len.cpu().long() + 1, batch_first=True)
        x_pred, _ = pad_packed_sequence(x_pred, batch_first=True)

        if args.weights:
            # Weights for BCE
            weight = torch.cat((feature_map['t1_weight'].to(args.device), feature_map['t2_weight'].to(args.device),
                                feature_map['v1_weight'].to(
                                    args.device), feature_map['e_weight'].to(args.device),
                                feature_map['v2_weight'].to(args.device)))

            weight = weight.expand(batch_size, x_len_max + 1, -1)
        else:
            weight = None

        loss_sum = F.binary_cross_entropy(
            x_pred, x_target, reduction='none', weight=weight)
        loss = torch.mean(
            torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))

    elif args.loss_type == 'NLL':
        timestamp1 = timestamp1.transpose(dim0=1, dim1=2)
        timestamp2 = timestamp2.transpose(dim0=1, dim1=2)
        vertex1 = vertex1.transpose(dim0=1, dim1=2)
        edge = edge.transpose(dim0=1, dim1=2)
        vertex2 = vertex2.transpose(dim0=1, dim1=2)

        loss_t1 = F.nll_loss(
            timestamp1, t1, ignore_index=max_nodes + 1, weight=feature_map.get('t1_weight'))
        loss_t2 = F.nll_loss(
            timestamp2, t2, ignore_index=max_nodes + 1, weight=feature_map.get('t2_weight'))
        loss_v1 = F.nll_loss(vertex1, v1, ignore_index=len(
            feature_map['node_forward']) + 1, weight=feature_map.get('v1_weight'))
        loss_e = F.nll_loss(edge, e, ignore_index=len(
            feature_map['edge_forward']) + 1, weight=feature_map.get('e_weight'))
        loss_v2 = F.nll_loss(vertex2, v2, ignore_index=len(
            feature_map['node_forward']) + 1, weight=feature_map.get('v2_weight'))

        loss = loss_t1 + loss_t2 + loss_v1 + loss_e + loss_v2

    return loss


def predict_graphs(eval_args):
    # 載入訓練參數與模型特徵對應資訊
    train_args = eval_args.train_args
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)
    train_args.device = eval_args.device

    # Whether the node types of graph are time-aware
    diff_node_type_time = eval_args.diff_node_type_time
    LIMIT_REPEAT = 5    # Maximum number of times to repeat the prediction step

    # 建立模型並載入已訓練參數
    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)

    for _, net in model.items():
        net.eval()

    # 計算 feature 向量長度
    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    graphs = []
    # graph_cnt = 3

    # mask_path = Pathlib.Path('./check_mask')
    # if mask_path.exists():
    #     for file in mask_path.iterdir():
    #         if file.is_file():
    #             file.unlink()
    # mask_path.mkdir(parents=True, exist_ok=True)

    if diff_node_type_time:
        postfixes = ['_early', '_mid', '_late']
        nb = feature_map['node_backward']
        postfix_mask = [np.array([False] * len(nb) + [True]) for _ in range(len(postfixes))]
        for i, node in nb.items():
            for j, postfix in enumerate(postfixes):
                if node.endswith(postfix):
                    postfix_mask[j][i] = True

                # mask_file = mask_path / f'mask{postfix}.csv'
                # with open(mask_file, 'a') as f:
                #     f.write(f'{i},{node},{postfix_mask[j][i]}\n')

    def pred_step(model, rnn_input):
        # rnn_output (h_i) = LSTM(h_{i-1}, Embed(s_{i-1}))
        rnn_output = model['dfs_code_rnn'](rnn_input)

        # Evaluating dfscode tuple
        # 模型輸出五個分支：timestamp1, timestamp2, vertex1, edge, vertex2
        # MLP layer (Linear + ReLU + Softmax)
        timestamp1 = model['output_timestamp1'](
            rnn_output).reshape(eval_args.batch_size, -1)
        timestamp2 = model['output_timestamp2'](
            rnn_output).reshape(eval_args.batch_size, -1)
        vertex1 = model['output_vertex1'](
            rnn_output).reshape(eval_args.batch_size, -1)
        edge = model['output_edge'](rnn_output).reshape(
            eval_args.batch_size, -1)
        vertex2 = model['output_vertex2'](
            rnn_output).reshape(eval_args.batch_size, -1)

        # 根據損失函數類型進行 sampling
        if train_args.loss_type == 'BCE':
            timestamp1 = Categorical(timestamp1).sample()
            timestamp2 = Categorical(timestamp2).sample()
            vertex1 = Categorical(vertex1).sample()
            edge = Categorical(edge).sample()
            vertex2 = Categorical(vertex2).sample()

        elif train_args.loss_type == 'NLL':
            timestamp1 = Categorical(logits=timestamp1).sample()
            timestamp2 = Categorical(logits=timestamp2).sample()
            vertex1 = Categorical(logits=vertex1).sample()
            edge = Categorical(logits=edge).sample()
            vertex2 = Categorical(logits=vertex2).sample()

        return timestamp1, timestamp2, vertex1, edge, vertex2

    # check_path = Pathlib.Path('./check_batch_fix')
    # if check_path.exists():
    #     for file in check_path.iterdir():
    #         if file.is_file():
    #             file.unlink()
    # check_path.mkdir(parents=True, exist_ok=True)

    # 每個 batch 產生一批圖
    with torch.no_grad():
        for _ in range(eval_args.count // eval_args.batch_size):
            # initialize dfs_code_rnn hidden according to batch size
            model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
                batch_size=eval_args.batch_size)

            # 初始化 RNN 輸入與輸出空間
            rnn_input = torch.zeros(
                (eval_args.batch_size, 1, feature_len), device=eval_args.device)
            pred = torch.zeros(
                (eval_args.batch_size, eval_args.max_num_edges, 5), device=eval_args.device)

            if diff_node_type_time:
                batch_sample_mask = np.array([False] * eval_args.batch_size)
                batch_sample_repeat_cnt = [0] * eval_args.batch_size
                tmp_rnn_input = torch.zeros(
                    (eval_args.batch_size, 1, feature_len), device=eval_args.device)
                
                batch_postfix_mask_i = [-1] * eval_args.batch_size

            # 每個 batch 逐步產生 DFS code 中的 edge tuples
            for step_i in tqdm(range(eval_args.max_num_edges), total=eval_args.max_num_edges, desc='Generating graphs (steps)'):
                
                if diff_node_type_time:
                    repeat_flag = True
                    for batch_sample_i in range(eval_args.batch_size):
                        if batch_sample_repeat_cnt[batch_sample_i] < LIMIT_REPEAT:
                            batch_sample_repeat_cnt[batch_sample_i] = 0     # Reset repeat count each step

                    while repeat_flag:
                        timestamp1, timestamp2, vertex1, edge, vertex2 = pred_step(model, rnn_input)

                        # Filter sampled vertices to include only nodes with the desired postfix
                        for batch_sample_i in range(eval_args.batch_size):

                            if batch_sample_repeat_cnt[batch_sample_i] >= LIMIT_REPEAT:    # Skip this sample (Not used for graph generation)
                                batch_sample_mask[batch_sample_i] = True

                            if batch_sample_mask[batch_sample_i]:
                                continue

                            # check_file = check_path / f'check_{batch_sample_i}.csv'
                            # nb = feature_map['node_backward']
                            # with open(check_file, 'a') as f:
                            #     v1 = nb.get(int(vertex1[batch_sample_i].data), 'End')
                            #     v2 = nb.get(int(vertex2[batch_sample_i].data), 'End')
                            #     f.write(f'{step_i},{v1},{v2},{int(vertex1[batch_sample_i].data)},{int(vertex2[batch_sample_i].data)}\n')

                            if batch_postfix_mask_i[batch_sample_i] == -1:
                                for postfix_i, mask in enumerate(postfix_mask):
                                    if mask[vertex1[batch_sample_i]] == True \
                                        and mask[vertex2[batch_sample_i]] == True \
                                        and batch_postfix_mask_i.count(postfix_i) <= eval_args.batch_size // len(postfix_mask) + 1:
                                        # If the sampled vertex1 and vertex2 has the desired postfix, set the mask
                                        batch_postfix_mask_i[batch_sample_i] = postfix_i
                                        batch_sample_mask[batch_sample_i] = True

                                        # 組合為下一步的暫存 RNN input（One-hot）
                                        tmp_rnn_input[batch_sample_i, 0, timestamp1[batch_sample_i]] = 1
                                        tmp_rnn_input[batch_sample_i,
                                                0, max_nodes + 1 + timestamp2[batch_sample_i]] = 1
                                        tmp_rnn_input[batch_sample_i,
                                                0, 2 * max_nodes + 2 + vertex1[batch_sample_i]] = 1
                                        tmp_rnn_input[batch_sample_i, 0,
                                                2 * max_nodes + 2 + len_node_vec + edge[batch_sample_i]] = 1
                                        tmp_rnn_input[batch_sample_i, 0, 2 *
                                                max_nodes + 2 + len_node_vec + len_edge_vec + vertex2[batch_sample_i]] = 1

                                        # 儲存 sample 結果
                                        pred[batch_sample_i, step_i, 0] = timestamp1[batch_sample_i]
                                        pred[batch_sample_i, step_i, 1] = timestamp2[batch_sample_i]
                                        pred[batch_sample_i, step_i, 2] = vertex1[batch_sample_i]
                                        pred[batch_sample_i, step_i, 3] = edge[batch_sample_i]
                                        pred[batch_sample_i, step_i, 4] = vertex2[batch_sample_i]
                                        
                                        break

                            else:
                                target_postfix_mask = postfix_mask[batch_postfix_mask_i[batch_sample_i]]

                                if target_postfix_mask[vertex1[batch_sample_i]] == True \
                                    and target_postfix_mask[vertex2[batch_sample_i]] == True:
                                    # If the sampled vertex1 and vertex2 has the desired postfix, set the mask
                                    batch_sample_mask[batch_sample_i] = True

                                    # 組合為下一步的暫存 RNN input（One-hot）
                                    tmp_rnn_input[batch_sample_i, 0, timestamp1[batch_sample_i]] = 1
                                    tmp_rnn_input[batch_sample_i,
                                            0, max_nodes + 1 + timestamp2[batch_sample_i]] = 1
                                    tmp_rnn_input[batch_sample_i,
                                            0, 2 * max_nodes + 2 + vertex1[batch_sample_i]] = 1
                                    tmp_rnn_input[batch_sample_i, 0,
                                            2 * max_nodes + 2 + len_node_vec + edge[batch_sample_i]] = 1
                                    tmp_rnn_input[batch_sample_i, 0, 2 *
                                            max_nodes + 2 + len_node_vec + len_edge_vec + vertex2[batch_sample_i]] = 1

                                    # 儲存 sample 結果
                                    pred[batch_sample_i, step_i, 0] = timestamp1[batch_sample_i]
                                    pred[batch_sample_i, step_i, 1] = timestamp2[batch_sample_i]
                                    pred[batch_sample_i, step_i, 2] = vertex1[batch_sample_i]
                                    pred[batch_sample_i, step_i, 3] = edge[batch_sample_i]
                                    pred[batch_sample_i, step_i, 4] = vertex2[batch_sample_i]

                        # if batch_sample_mask.sum() >= eval_args.batch_size // 3:
                        if batch_sample_mask.all():

                            # 組合為下一步的 RNN input（One-hot）
                            rnn_input = tmp_rnn_input.clone().detach()
                            batch_sample_mask = np.array([False] * eval_args.batch_size)
                            tmp_rnn_input = torch.zeros(
                                (eval_args.batch_size, 1, feature_len), device=eval_args.device)

                            repeat_flag = False

                        # print('batch_sample_mask:\n', batch_sample_mask)
                        # print()

                        else:
                            for idx in np.where(~batch_sample_mask)[0]:
                                batch_sample_repeat_cnt[idx] += 1

                else:
                    timestamp1, timestamp2, vertex1, edge, vertex2 = pred_step(model, rnn_input)

                    # 組合為下一步的 RNN input（One-hot）
                    rnn_input = torch.zeros(
                        (eval_args.batch_size, 1, feature_len), device=eval_args.device)

                    rnn_input[torch.arange(eval_args.batch_size), 0, timestamp1] = 1
                    rnn_input[torch.arange(eval_args.batch_size),
                            0, max_nodes + 1 + timestamp2] = 1
                    rnn_input[torch.arange(eval_args.batch_size),
                            0, 2 * max_nodes + 2 + vertex1] = 1
                    rnn_input[torch.arange(eval_args.batch_size), 0,
                            2 * max_nodes + 2 + len_node_vec + edge] = 1
                    rnn_input[torch.arange(eval_args.batch_size), 0, 2 *
                            max_nodes + 2 + len_node_vec + len_edge_vec + vertex2] = 1

                    # 儲存 sample 結果
                    pred[:, step_i, 0] = timestamp1
                    pred[:, step_i, 1] = timestamp2
                    pred[:, step_i, 2] = vertex1
                    pred[:, step_i, 3] = edge
                    pred[:, step_i, 4] = vertex2

                    # for batch_sample_i in range(eval_args.batch_size):
                    #     check_file = check_path / f'check_{batch_sample_i}.csv'
                    #     nb = feature_map['node_backward']
                    #     with open(check_file, 'a') as f:
                    #         v1 = nb.get(int(vertex1[batch_sample_i].data), 'End')
                    #         v2 = nb.get(int(vertex2[batch_sample_i].data), 'End')
                    #         f.write(f'{step_i},{v1},{v2},{int(vertex1[batch_sample_i].data)},{int(vertex2[batch_sample_i].data)}\n')

            # 將 DFS code 轉換為 NetworkX graph
            nb = feature_map['node_backward']
            eb = feature_map['edge_backward']
            for i in range(eval_args.batch_size):
                if batch_sample_repeat_cnt[i] >= LIMIT_REPEAT:   # Skip this sample (Not used for graph generation)
                    continue

                dfscode = []
                for j in range(eval_args.max_num_edges):
                    if pred[i, j, 0] == max_nodes or pred[i, j, 1] == max_nodes \
                            or pred[i, j, 2] == len_node_vec - 1 or pred[i, j, 3] == len_edge_vec - 1 \
                            or pred[i, j, 4] == len_node_vec - 1:
                        break

                    dfscode.append(
                        (int(pred[i, j, 0].data), int(pred[i, j, 1].data), nb[int(pred[i, j, 2].data)],
                        eb[int(pred[i, j, 3].data)], nb[int(pred[i, j, 4].data)]))

                graph = graph_from_dfscode(dfscode)

                # Remove self loops
                graph.remove_edges_from(nx.selfloop_edges(graph))

                # Take maximum connected component
                if len(graph.nodes()):
                    max_comp = max(nx.connected_components(graph), key=len)
                    graph = nx.Graph(graph.subgraph(max_comp))

                # if len(graph.nodes()) > 0:
                # print('dfscode:', dfscode)  ######
                # print('original graph:')
                # print(graph.nodes(data=True))
                # print(graph.edges(data=True))
                # print('\n\n')
                # graph_cnt -= 1
                # if graph_cnt == 0:
                #     exit()

                graphs.append(graph)

    return graphs
