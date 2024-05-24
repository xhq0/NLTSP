import pickle
import torch
import logging
from .cost_model import PythonBasedModel
from tvm.auto_scheduler.feature import get_per_store_features_from_states_nltsp
from torch import nn
import numpy as np
logger = logging.getLogger("auto_scheduler")

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


class NLTSPModel(PythonBasedModel):
    """The wrapper of MLPModelInternal. So we can use it in end-to-end search."""
    def __init__(self, target, num_axes=12):
        super().__init__()
        self.num_axes = num_axes

    def update(self, inputs, results):
        pass

    def predict(self, task, states):
        segment_sizes, features = get_per_store_features_from_states_nltsp(states, task)
        features.resize(np.sum(segment_sizes), self.num_axes, 9)
        
        lx = 12 # IteratorAnnotation, one-hot encoding of length 12
        ly = 4  # IteratorKind , one-hot encoding of length 4
        lz = 51 # op_name , one-hot encoding of length 51
        classes = features[:, :, :3].astype(int)
        xi = np.arange(len(features))[:, np.newaxis] 
        yi = np.arange(len(features[0]))

        feature_tensor = torch.zeros((np.sum(segment_sizes), self.num_axes, 73), dtype=torch.float32, device='cuda:0')
        feature_tensor[xi, yi, classes[:, :, 2] + (6 + lx + ly - 1)] = 1
        feature_tensor[xi, yi, classes[:, :, 1] + (6 + lx - 1)] = 1
        feature_tensor[xi, yi, classes[:, :, 0] + (6 - 1)] = 1
        feature_tensor[:,:,:6] = torch.tensor(features[:, :, 3:], device="cuda:0")
        
        segment_sizes = torch.tensor(segment_sizes, dtype=torch.long, device="cuda:0")

        with torch.no_grad():
            ret = self.model(feature_tensor, segment_sizes)
        if isinstance(ret, list) and len(ret) > 0:
            ret = ret[0]
        return ret.cpu().detach().numpy()


    def load(self, file_name: str):
        self.model = LSTMModule(
            fea_size = 9 - 3 + 12 + 4 + 51, 
            step_size = self.num_axes
        ).to('cuda:0')

        state_dict = torch.load(file_name)
        new_state_dict = {key.split("module.", 1)[-1]: value if key.startswith("module.") else value for key, value in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

