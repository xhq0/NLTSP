# NLTSP: A Cost Model for Tensor Program Tuning Using Nested Loop Trees

This repo is based on a fork of [tenset](https://github.com/tlc-pack/tenset).

## Installation

This repo is built upon TVM, and you can install this repo in a similar manner to installing TVM from source. For installation instructions, please refer to the following link: [Install TVM from Source](https://tvm.apache.org/docs/install/from_source.html).


Version information can refer to [here](version.log).

## Download the TenSet datasets

1. Download

   You can download [tenset_cpu_v3.3.zip](https://drive.google.com/file/d/1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6/view?usp=sharing), [tenset_gpu_v3.3.zip](https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view?usp=sharing) from google drive. And put these zip files under `nltsp/scripts`.

2. Unzip

   ```shell
   cd scripts
   unzip dataset_cpu_v3.3.zip
   unzip dataset_gpu_v3.3.zip
   ```

## Train a NLTSP cost model

### 1. Make a Dataset

To generate a dataset, follow these steps:

- Open `nltsp_make_dataset.sh`.
- Set `json_files_path` to the directory containing tenset measurement data, e.g., `dataset/measure_records/t4/`.
- Set `output_path` to the desired output directory for the dataset.
- For CPU datasets, set `platform="llvm"`; for GPU datasets, set `platform="cuda"`.
- To create a training set, set `dataset_type="train"`; to create a test set, set `dataset_type="test"`.
- Adjust `files_cnt` to modify the number of tasks in the training set.
- Modify `sampling_cnt` to change the number of measurement records per task.

Then execute:

```shell
bash nltsp_make_dataset.sh
```

### 2. Train. 

To train the model, follow these steps:

- Open `nltsp_run.sh`.
- Set `job="train"`, `dataset` to the path of the training set, and `output_path` to the cost model's output directory.
- Specify the GPU device number in `device`.
- Execute:

```shell
bash nltsp_run.sh
```
   You can adjust other parameters in nltsp_run.sh to experiment with different learning rates, weight decay factors, etc.

### 3. Eval

To evaluate the model, follow these steps:

- Open nltsp_run.sh.
- Set `job="eval"`, `dataset` to the path of the test set, and `trained_model` to the path of the trained cost model.
- If it's a CPU task, set `platform="llvm"`; if it's a GPU task, set `platform="cuda"`.
- Execute:

   ```shell
   bash nltsp_run.sh
   ```
These steps will evaluate the model's performance on the test set.


## Use the model for search


```shell
python tune_network.py \
   --network "bert_base" \
   --n-trials 2000 \
   --cost-model nltsp-no-update \
   --load-model trained_model/nltsp-t4/nltsp_model_25.pkl \
   --target 'cuda -model=t4'  \
   --num_measures_per_round 10 \
```
`network` refers to the name of the deep learning workload, and `load-model` denotes the path to the pre-trained cost model. For GPU tasks, the target should start with `cuda`, while for CPU tasks, the target should start with `llvm`.

## License
The code is licensed under an [Apache-2.0](LICENSE) license.  
