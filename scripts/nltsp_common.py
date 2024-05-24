import argparse
import logging
import os
import torch
import torch.cuda
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--platform", type=str, default="cuda", choices=["cuda", "llvm"])
    parser.add_argument("--trained_model", type=str, default="None",help="A trained model for testing.")
    parser.add_argument("--dataset", type=str, help=".pkl used for training")
    parser.add_argument("--output_path",type=str,default="None")

    parser.add_argument("--device",type=int,default=0,help="Computation device id, -1 for cpu")
    parser.add_argument("--num_workers",type=int,default=1,help="A parameter of GraphDataLoader",)

    parser.add_argument("--model_name", type=str, default="tree_lstm")
    parser.add_argument("--axis_fea_size", type=int, default=0) 
    parser.add_argument("--num_axes", type=int, default=0) 

    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay rate")
    parser.add_argument("--gamma", type=float, default=0.95) # 指数衰减因子 gamma
    parser.add_argument("--scheduler_step_size", type=int, default=1000)
        
    args = parser.parse_args()
        
    # device
    args.device = "cpu" if args.device < 0 else "cuda:{}".format(args.device)
    if not torch.cuda.is_available():
        logging.warning("GPU is not available, using CPU for training")
        args.device = "cpu"
    else:
        logging.warning("Device: {}".format(args.device))
    
    paths = [args.output_path]
    for p in paths:
        if p == "None":
            continue
        if not os.path.exists(p):
            os.makedirs(p)

    return args




class LambdaRankLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1.):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :,
                                          None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (
            y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(
            sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss

