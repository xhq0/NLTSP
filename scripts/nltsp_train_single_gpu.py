import os
os.environ["DGLBACKEND"] = "pytorch"
import pickle
import torch 
from torch.optim import Adam
import numpy as np
import random
import time
from nltsp_common import parse_args, LambdaRankLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn


class LSTMModule(nn.Module):
    def __init__(self,fea_size, step_size):
        super().__init__()
        self.fea_size = fea_size
        self.step_size = step_size
        lstm_linar_in_dim = self.fea_size
        lstm_linar_hidden_dim = [64, 256]
        out_dim = [256, 64, 1]
        hidden_dim = lstm_linar_hidden_dim[-1]
        self.lstm_linar_encoder = nn.Sequential(
            nn.Linear(lstm_linar_in_dim, lstm_linar_hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[0], lstm_linar_hidden_dim[1]),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            lstm_linar_hidden_dim[-1], lstm_linar_hidden_dim[-1])
        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
        )

    def forward(self, batch_datas_steps, segment_sizes):
        device = batch_datas_steps.device
        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        batch_datas_steps = batch_datas_steps.transpose(0, 1)
        lstm_output = self.lstm_linar_encoder(batch_datas_steps)
        _, (h, c) = self.lstm(lstm_output)
        lstm_output = h[0]
        output = lstm_output
        n_seg = segment_sizes.shape[0]
        segment_indices = torch.repeat_interleave(
            torch.arange(n_seg, device=device), segment_sizes
        )
        n_dim = output.shape[1] 
        output = torch.scatter_add(
            torch.zeros((n_seg, n_dim), dtype=output.dtype, device=device),
            0,
            segment_indices.view(-1, 1).expand(-1, n_dim),
            output,
        )
        output = self.l0(output) + output
        output = self.l1(output) + output
        output = self.decoder(output)
        return output.squeeze()


class NLTSPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        m = self.data[idx]
        return m[0], len(m[0]), m[1] # feature, segment_size, label

    def statistics(self):
        m = self.data[0]
        return len(m[0][0][0])

    def get_labels(self):
        return [i[1] for i in self.data]

    def get_min_cost(self):
        min_costs = [i[2] for i in self.data]
        assert min(min_costs) == max(min_costs)
        return min_costs[0]


def collate(samples):
    features, segment_sizes, labels = map(list, zip(*samples))
    features = np.concatenate(features, axis=0)

    lx = 12 # IteratorAnnotation, one-hot encoding of length 12
    ly = 4  # IteratorKind , one-hot encoding of length 4
    lz = 51 # op_name , one-hot encoding of length 51
    classes = features[:, :, :3].astype(int)
    xi = np.arange(len(features))[:, np.newaxis] 
    yi = np.arange(len(features[0]))
    
    feature_tensor = torch.zeros((len(features), 12, 73), dtype=torch.float32)
    feature_tensor[xi, yi, classes[:, :, 2] + (6 + lx + ly - 1)] = 1
    feature_tensor[xi, yi, classes[:, :, 1] + (6 + lx - 1)] = 1
    feature_tensor[xi, yi, classes[:, :, 0] + (6 - 1)] = 1
    feature_tensor[:,:,:6] = torch.tensor(features[:, :, 3:])

    segment_sizes = torch.tensor(np.array(segment_sizes, dtype=np.int64))
    labels = torch.tensor(np.array(labels, dtype=np.float32))
    return feature_tensor, segment_sizes, labels



def get_dataloaders(dataset, seed, parser_args):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, 
        batch_size = parser_args.batch_size, 
        shuffle = True, 
        num_workers = parser_args.num_workers, 
        collate_fn = collate, 
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size = parser_args.batch_size,
        num_workers = parser_args.num_workers, 
        collate_fn = collate
    )

    return train_loader, val_loader


def init_model(seed, device, parser_args):
    if parser_args.model_name == "lstm":
        model = LSTMModule(
            fea_size = parser_args.axis_fea_size, 
            step_size = parser_args.num_axes
        ).to(device)
    else:
        raise ValueError("Invalid model name: {}".format(parser_args.model_name))
    return model


def run(rank, world_size, dataset, parser_args, seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    model = init_model(seed, device, parser_args)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    optimizer = Adam(model.parameters(), lr = parser_args.lr, amsgrad=True, weight_decay=parser_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = parser_args.scheduler_step_size, gamma = parser_args.gamma)
    criterion = LambdaRankLoss(device)
    train_loader, val_loader = get_dataloaders(dataset, seed, parser_args)

    for epoch in range(parser_args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for features, segment_sizes, labels in tqdm(train_loader, desc='Training'):
            segment_sizes = segment_sizes.to(device) 
            labels = labels.to(device) 
            features = features.to(device)



            if parser_args.model_name in ["lstm"]:
                loss = criterion(model(features, segment_sizes), labels)
            else:
                raise ValueError("Invalid model name: {}".format(parser_args.model_name))

            total_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss = total_loss
        print("Loss: {:.4f}".format(loss))
        
        if(rank == 0):
            model_save_file_name = '%s/nltsp_model_%d.pkl' % (parser_args.output_path, epoch)
            torch.save(model.state_dict(), model_save_file_name)
            print("Model %d has been saved." % epoch)
            print("epoch {} took {} s".format(epoch, time.time() - start_time))




if __name__ == "__main__":
    parser_args = parse_args()
    # Create a save directory
    if not os.path.exists(parser_args.output_path):
        os.makedirs(parser_args.output_path)

    # Check whether GPU is available.
    if not torch.cuda.is_available():
        print("No GPU found!")
        exit()

    num_gpus = torch.cuda.device_count()
    with open(parser_args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    
    nltsp_dataset = NLTSPDataset(dataset)    

    parser_args.axis_fea_size = nltsp_dataset.statistics() - 3 + 12 + 4 + 51
    print("parser_args.axis_fea_size = ",parser_args.axis_fea_size)
    run(0, num_gpus , nltsp_dataset, parser_args, seed = 42)