import os
os.environ["DGLBACKEND"] = "pytorch"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import pickle
import numpy as np
import torch
from nltsp_common import parse_args
from nltsp_train_single_gpu import init_model, collate
from tqdm import tqdm
import pickle
import torch 
import sys


class NLTSPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, slice_obj):

        t = self.data[slice_obj]
        return [[item[0], len(item[0]), item[1]] for item in t]

    def statistics(self): # Dimension of feature
        m = self.data[0]
        return len(m[0][0])

    def get_labels(self):
        return [i[1] for i in self.data]

    def get_min_cost(self):
        min_costs = [i[2] for i in self.data]
        assert min(min_costs) == max(min_costs)
        return min_costs[0]


def vec_to_pair_com(vec):
    return (vec.reshape((-1, 1)) - vec) > 0


def metric_pairwise_comp_accuracy(preds, labels):
    """Compute the accuracy of pairwise comparision"""
    n = len(preds)
    if n <= 1:
        return 0.5
    preds = vec_to_pair_com(preds)
    labels = vec_to_pair_com(labels)
    correct_ct = np.triu(np.logical_not(np.logical_xor(preds, labels)), k=1).sum()
    return correct_ct / (n * (n-1) / 2)

top_ks = [1, 5, 10, 20]

    
def pred_a_dataset(datas, model, parser_args):
    # model: Trained model
    device = torch.device(parser_args.device)

    nltsp_dataset = NLTSPDataset(datas)
    
    model.eval()
    preds_list = []

    # batch_size = 2
    batch_size = parser_args.batch_size
    num_batches = len(nltsp_dataset) // batch_size
    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(nltsp_dataset))
        samples = nltsp_dataset[start_idx:end_idx]

        features, segment_sizes, labels = collate(samples)

        
        segment_sizes = segment_sizes.to(device) 
        features = features.to(device) 

        if parser_args.model_name == "lstm":
            batch_preds = model(features, segment_sizes)
        else:
            raise ValueError("Invalid model name: {}".format(parser_args.model_name))

        preds_list.append(batch_preds)
    preds = torch.cat(preds_list).detach().cpu().numpy()

    return (preds, nltsp_dataset.get_min_cost(), np.array(nltsp_dataset.get_labels()))
    

    


def eval_model(parser_args):
    device = torch.device(parser_args.device)
    model = init_model(42, device, parser_args )

    # Read in model parameters
    state_dict = torch.load(parser_args.trained_model)
    new_state_dict = {key.split("module.", 1)[-1]: value if key.startswith("module.") else value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    subgraph = {}
    for data_idx, data in enumerate(test_datasets):
        file, file_idx, _, workloadkey, workload_args, flop_ct, line_vecs = data
        subgraph[workloadkey] = line_vecs

    files = [
        'dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl' % parser_args.platform,
        'dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl' % parser_args.platform,
        'dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl' % parser_args.platform,
        'dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl' % parser_args.platform,
        'dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl' % parser_args.platform
    ]
    
    best_latency_total_list, total_weights, total_pair_acc_list = [], [], [] 
    best_latency_total, top1_total, top5_total, top10_total, top20_total = 0, 0, 0, 0, 0

    for file in files:
        tasks, task_weights = pickle.load(open(file, "rb")) 
        latencies = [0] * len(top_ks)
        best_latency = 0
        flag = 0
        pair_acc_list = []

        for task, weight in zip(tasks, task_weights): 
            if task.workload_key not in subgraph:
                print('error task.workload_key not in subgraph')
                continue
            flag = 1
            
            preds, min_latency, labels = pred_a_dataset(subgraph[task.workload_key], model, parser_args)
            pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))


            real_values = labels[np.argsort(-preds)] 
            real_latency = min_latency / np.maximum(real_values, 1e-5) 

            for i, top_k in enumerate(top_ks): # top_ks = [1, 5, 10, 20]
                latencies[i] += np.min(real_latency[:top_k]) * weight
            best_latency += min_latency * weight
        if flag == 0:
            continue

        print(f"top 1 score: {best_latency/latencies[0]}")
        print(f"top 5 score: {best_latency / latencies[1]}")
        pair_acc = np.average(pair_acc_list, weights=task_weights)
        print(f"pair_acc: {pair_acc}")

        total_weights.extend(task_weights)
        total_pair_acc_list.extend(pair_acc_list)

        best_latency_total_list.append(best_latency)
        best_latency_total += best_latency
        top1_total += latencies[0]
        top5_total += latencies[1]
        top10_total += latencies[2]
        top20_total += latencies[3]

    
    print(f"average top 1 score is {best_latency_total / top1_total}")
    print(f"average top 5 score is {best_latency_total / top5_total}")
    print(f"average top 10 score is {best_latency_total / top10_total}")
    print(f"average top 20 score is {best_latency_total / top20_total}")
    total_pair_acc = np.average(total_pair_acc_list, weights=total_weights)
    print(f"average pair_acc: {total_pair_acc}")

if __name__ == "__main__":
    parser_args = parse_args()
    with open(parser_args.dataset, 'rb') as f:
        test_datasets = pickle.load(f)

    # Get the length of the feature vector
    # Subgraph 0 (95 in total)
    # 6 Subgraph Features (7 in total)
    # The 0th schedule (4000 in total)
    # The 0th scheduling feature (4 in total)

    parser_args.axis_fea_size = len(test_datasets[0][6][0][0][2][0]) - 3 + 12 + 4 + 51
    print("parser_args.axis_fea_size = ",parser_args.axis_fea_size)
    
    eval_model(parser_args)