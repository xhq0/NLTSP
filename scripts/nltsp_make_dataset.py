import os
import json
import pickle
from tvm import auto_scheduler
from common import load_and_register_tasks
import threading
import multiprocessing
from tvm.tir.expr import FloatImm
import numpy as np
import random
import argparse
import math
from tvm.auto_scheduler.feature import get_per_store_features_from_states_nltsp


def handle_file(file_idx, file):
    try:
        inputs, outputs = auto_scheduler.RecordReader(file).read_lines()
    except Exception:
        print(f"Error reading file: {file_idx}-{file}")
        return None
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task

    workload_args = [int(i) for i in task.workload_key[len('["6b7583cf23c7c37d3212cad9d06e58c1", '): -1].split(', ')]
    line_vecs = []
    min_cost = 1000000

    # schedule_cnt = len(lines) # 一般来说，是4000
    schedule_cnt = len(inputs)

    if schedule_cnt > 4000:
        raise ValueError("schedule_cnt should not exceed 4000")
    if args.dataset_type == "train":
        sampling_cnt = min(args.sampling_cnt, schedule_cnt)
    elif args.dataset_type == "test":
        sampling_cnt = schedule_cnt

    print(file_idx, sampling_cnt)
    sampling_list = random.sample(range(schedule_cnt), sampling_cnt)

    for line_idx in sampling_list:
        measure_inp = auto_scheduler.measure.recover_measure_input(inputs[line_idx], True)
        state = measure_inp.state
        task = measure_inp.task

        segment_sizes, features = get_per_store_features_from_states_nltsp([state], task)
        features.resize(np.sum(segment_sizes), 12, 9)
        costs = [x.value for x in outputs[line_idx].costs if isinstance(x, FloatImm)]
        cost = np.mean(costs)
        min_cost = min(min_cost, cost)

        line_vecs.append([features, cost, None])

    for idx in range(len(line_vecs)):
        line_vecs[idx][2] = min_cost
        line_vecs[idx][1] = min_cost / line_vecs[idx][1]
    
    if args.dataset_type == "train":
        return line_vecs
    elif args.dataset_type == "test":
        return [file, file_idx, None, task.workload_key, workload_args, task.compute_dag.flop_ct, line_vecs]

def make_all_dataset():
    tasks = load_and_register_tasks()

    json_record = f"json_{args.platform}_{args.dataset_type}.txt"
    with open(json_record, 'r') as file:
        lines = file.readlines()
        json_files = [args.json_files_path + line.strip() for line in lines]

    if args.dataset_type == "train":
        json_files = random.sample(json_files, args.files_cnt)

    multiprocessing_pool = multiprocessing.Pool(processes = 64)
    que_res_list = []
    for file_idx, file in enumerate(json_files):
        que_res_list.append(multiprocessing_pool.apply_async(handle_file, args=(file_idx, file)))
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    if args.dataset_type == "train":
        file_vecs = []
        for que_res in que_res_list:
            response = que_res.get()
            if response:
                file_vecs.extend(response)
    elif args.dataset_type == "test":
        file_vecs = [que_res.get() for que_res in que_res_list]

    return file_vecs


def save_dataset(file_vecs):
    hardware_latforms = os.path.basename(args.json_files_path[:-1])
    filename = f"nltsp_dataset_{hardware_latforms}_{args.files_cnt}_{args.sampling_cnt}_{args.dataset_type}.pkl"
    with open(args.output_path + filename, 'wb') as f:
        pickle.dump(file_vecs, f)
    

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_files_path", type=str, default='dataset/measure_records/platinum-8272')
    parser.add_argument("--files_cnt", type=int)
    parser.add_argument("--platform", type=str, choices=['llvm', 'cuda']) 
    parser.add_argument("--sampling_cnt", type=int, default=4000) 
    parser.add_argument("--dataset_type", type=str, choices=['train', 'test']) 
    parser.add_argument("--output_path", type=str) 
    args = parser.parse_args()


    hold_out_tasks = []
    files = [
        'dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl',
        'dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl'
    ]
    for file in files:
        tasks_part, task_weights = pickle.load(open(file % args.platform, "rb"))
        hold_out_tasks.extend(tasks_part)
    hold_out_tasks_set = set([task.workload_key for task in hold_out_tasks])

    
    file_vecs = make_all_dataset()
    save_dataset(file_vecs)
    print('make dataset nltsp done.')



