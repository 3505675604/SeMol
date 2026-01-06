#coding=utf-8
import os
import pickle
import argparse
import pandas as pd
import torch
import torch_geometric
from torch import nn
from model.transformer_model import transformer_1d
import numpy as np
from model.feature_fussion import TransformerEncoder
from model.gnn_model import GNN,GNNDecoder
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
import torch.multiprocessing
from tqdm import tqdm
import torch.nn.functional as F
from utils import to_dense_with_fixed_padding
from process_dataset.MPP.utils.dist import init_distributed_mode
from process_dataset.MPP.data.DCGraphPropPredDataset.dataset import DCGraphPropPredDataset
from process_dataset.MPP.utils.evaluate import Evaluator
import random

from torch_geometric.nn import global_mean_pool
from model.SemMol import SemMol

np.set_printoptions(threshold=np.inf)

# device_ids = [1, 2]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE=256 * len(device_ids)
EPOCH = 1
SAVE_MODEL = '/data/FL/Semol/save_model/model_best.pth'


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attn_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        scores = self.attn_weights(x).squeeze(-1)
        scores[mask == 0] = -1e9

        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        if torch.isnan(attn_weights).any():
            print("Tensor contains NaN values!")

            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = (attn_weights * x).sum(dim=1)

        return context


def masked_average(x, mask):
    mask = mask.unsqueeze(-1)
    x_masked = x * mask
    valid_counts = mask.sum(dim=1).clamp(min=1)
    avg_result = x_masked.sum(dim=1) / valid_counts

    return avg_result


def masked_sum(x, mask):
    mask = mask.unsqueeze(-1)
    x_masked = x * mask
    sum_result = x_masked.sum(dim=1)

    return sum_result

class property_predictor(nn.Module):
    def __init__(self, model_state_dict,input_dim, hidden_dim, output_dim, device, dropout=0.5):
        super(property_predictor, self).__init__()
        self.attn_pooling = AttentionPooling(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pre = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.my_model = MyModel(device)
        self.my_model.load_state_dict(model_state_dict, strict=False)

        for name, p in self.my_model.named_parameters():
            if "feature_fussion" in name or "semmol" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False


        self.model_no_pretrain = MyModel(device)


    def forward(self, batch_data, aggre ='attn'):
        fussion_feature, valid_position = self.my_model(batch_data)

        original_fussion_feature, original_valid_position = self.model_no_pretrain(batch_data)
        if aggre == 'attn':
            x = self.attn_pooling(fussion_feature,valid_position)  # [batch_size, 128]
            original_x = self.attn_pooling(original_fussion_feature,original_valid_position)
        elif aggre == 'mean':
            x = masked_average(fussion_feature,valid_position)
            original_x = masked_average(original_fussion_feature, original_valid_position)
        elif aggre == 'sum':
            x = masked_sum(fussion_feature, valid_position)
            original_x = masked_sum(original_fussion_feature,original_valid_position)

        emd=x
        original_emd = original_x
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  
        x = self.pre(x)

        return x,emd,original_emd
class PreprocessBatch:
    def process(self, batch):
        if (not hasattr(batch, "pos")) or (batch.pos is None):
            return
        pos = batch.pos
        pos_mean = global_mean_pool(pos, batch.batch)
        num_tensor = torch.bincount(batch.batch).to(pos.device)
        pos = pos - torch.repeat_interleave(pos_mean, num_tensor, dim=0)
        batch.pos = pos


class MyModel(nn.Module):
    def __init__(self, device):
        super(MyModel, self).__init__()
        self.device = device

        self.encoder_1d = transformer_1d()

        self.encoder_2d = GNN(num_layer=3, hidden_dim=128, output_dim=128)

        self.encoder_3d = DimeNet(
            hidden_channels=128,
            num_blocks=3,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            out_channels=128
        )

        self.semmol = SemMol(
            d=128, senter_number=50, TopK=5,
            anchor_mode='1d',
            use_bidirectional=False,
            use_cross_attention=False
        )

        self.feature_fussion = TransformerEncoder(128, 128, 8, 4)

        self.token_bias = nn.Parameter(torch.randn(50, 128))
        self.graph_bias = nn.Parameter(torch.randn(50, 128))
        self.molecule_bias = nn.Parameter(torch.randn(50, 128))

        self.preprocessor = PreprocessBatch()

    def forward(self, batch_data):
        batch_data = batch_data.to(self.device)
        self.preprocessor.process(batch_data)

        batch_size = batch_data.num_graphs

        tokens_emb = torch.tensor(np.array(batch_data.tokens), dtype=torch.long, device=self.device)
        smi_mask = torch.tensor(np.array(batch_data.attention_mask), dtype=torch.bool, device=self.device)

        token_representation_1d = self.encoder_1d(tokens_emb, smi_mask)  # [B, 50, 128]
        mask_1d = torch.ones(batch_size, 50, dtype=torch.bool, device=self.device)

        node_representation_2d = self.encoder_2d(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
        node_representation_2d, mask_2d = to_dense_with_fixed_padding(node_representation_2d, batch_data.batch, 50)

        if (hasattr(batch_data, "pos")) and (batch_data.pos is not None):
            node_representation_3d = self.encoder_3d(batch_data.x[:, 0].long(), batch_data.pos, batch_data.batch)
            node_representation_3d, mask_3d = to_dense_with_fixed_padding(node_representation_3d, batch_data.batch, 50)
        else:
            node_representation_3d = torch.zeros(batch_size, 50, 128, device=self.device)
            mask_3d = torch.zeros(batch_size, 50, dtype=torch.bool, device=self.device)


        token_representation_1d, node_representation_2d, node_representation_3d, _, _ = \
            self.semmol(token_representation_1d, node_representation_2d, node_representation_3d)


        token_representation_1d = token_representation_1d + self.token_bias.unsqueeze(0).expand(batch_size, -1, -1)
        node_representation_2d = node_representation_2d + self.graph_bias.unsqueeze(0).expand(batch_size, -1, -1)
        node_representation_3d = node_representation_3d + self.molecule_bias.unsqueeze(0).expand(batch_size, -1, -1)

        emd_sum = torch.cat([token_representation_1d, node_representation_2d, node_representation_3d], dim=1)
        mask_label = torch.cat([mask_1d, mask_2d, mask_3d], dim=1)

        fussion_feature = self.feature_fussion(emd_sum, mask_label)

        return fussion_feature, mask_label

def finetune_train(train_loader, cls_predictor, optimizer,tag, epoch, target_per_class=70):
    step = 0
    total_loss = 0
    cls_predictor.train()


    for step, batch_data in enumerate(tqdm(train_loader, desc=tag)):
        batch_data = batch_data.to(device)
        if batch_data.x.shape[0] == 1 or batch_data.batch[-1] == 0:
            pass
        else:
            pred_result,emd,original_emd = cls_predictor(batch_data)
            valid_label = batch_data.y == batch_data.y
            loss = F.binary_cross_entropy_with_logits(
                pred_result.to(torch.float32)[valid_label], batch_data.y.to(torch.float32)[valid_label])

            if torch.isnan(loss).any():
                print("Loss contains NaN!")
                break
            total_loss +=loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / (step + 1)




def finetune_evaluate(valid_loader,cls_predictor,evaluator,tag):
    valid_loader = valid_loader
    valid_data_len = len(valid_loader)
    cls_predictor.eval()
    y_true = []
    y_pred = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(valid_loader, desc=tag)):
            batch_data = batch_data.to(device)
            try:
                pred_result,_,_ = cls_predictor(batch_data)
                y_true.append(batch_data.y.view(pred_result.shape).detach().cpu())
                y_pred.append(pred_result.detach().cpu())
                total_preds = torch.cat((total_preds, pred_result.cpu()), 0)
                total_labels = torch.cat((total_labels, batch_data.y.cpu()), 0)
            except:
                continue
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="dc-bbbp")
    parser.add_argument("--num_class", type=int, default="1")
    parser.add_argument("--hidden_size", type=int, default="64")
    
    parser.add_argument("--learning_rate", type=float, default="1e-5")

    args = parser.parse_args()
    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # init_distributed_mode(args)
    print(args)

    if args.dataset.startswith("dc"):
        dataset  = DCGraphPropPredDataset(args.dataset)

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset, dataset=dataset)


    train_loader = torch_geometric.loader.DataLoader(
        dataset[split_idx["train"]], num_workers=args.num_workers,batch_size=args.batch_size,shuffle=False,
    )
    valid_loader = torch_geometric.loader.DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    seeds = [42]

    for seed in seeds:
        set_seed(seed)
        checkpoint = torch.load(SAVE_MODEL, map_location=device)
        model_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("module."):
                model_state_dict[k[len("module."):]] = v
            else:
                model_state_dict[k] = v
        cls_predictor = property_predictor(model_state_dict,128,64,args.num_class,device).to(device)
        optimizer = torch.optim.Adam(cls_predictor.parameters(),lr=args.learning_rate*3, weight_decay=1e-4)
        valid_curve = []
        test_curve = []
        best_val_epoch = 0
        best_val_auc = 0
        patience = 7
        counter = 0

        for epoch in range(1, args.epochs + 1):
            print("Epoch {}".format(epoch))
            print("Training...")
            train_loss= finetune_train(train_loader, cls_predictor, optimizer,"Training",epoch)
            print('Epoch {}, train loss: {:.4f}'.format(epoch, train_loss))
            print("Evaluating...")
            valid_perf = finetune_evaluate(valid_loader, cls_predictor, evaluator,"Validating")
            test_perf = finetune_evaluate(test_loader, cls_predictor, evaluator,"Testing")
            print("Validation", valid_perf['rocauc'], "Test", test_perf['rocauc'])
            val_auc = valid_perf[dataset.eval_metric]
            test_auc = test_perf[dataset.eval_metric]
            valid_curve.append(val_auc)
            test_curve.append(test_auc)

            now_best_val_epoch = np.argmax(np.array(valid_curve))
            if best_val_epoch != now_best_val_epoch:
                best_val_epoch = now_best_val_epoch

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                counter = 0
            else:
                counter+=1

            if counter >= patience:
                print("Early stopping triggered")
                break

        best_val_epoch = np.argmax(np.array(valid_curve))

        print("Finished training!")
        test_perf = finetune_evaluate(test_loader, cls_predictor, evaluator,"Testing")
        print(dataset.eval_metric)
        print("Test score: {}".format(test_perf[dataset.eval_metric]))


if __name__ == "__main__":
    main()


'''
"dc-hiv" 
"dc-bace"
"dc-clintox"
"dc-bbbp"
"dc-clearance"
"dc-delaney"
"dc-qm7"
"dc-qm8"
"dc-sider"
"dc-tox21"
"dc-toxcast"
"dc-muv"
"dc-qm9"
'''