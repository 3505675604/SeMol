#coding=utf-8
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch_geometric\.nn\.data_parallel"
)


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import argparse
import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
from pcqm4m import PCQM4Mv2Dataset
from model.transformer_model import transformer_1d
import numpy as np
from model.SemMol import SemMol
from model.gnn_model import GNN, GNNDecoder
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
from model.feature_fussion import TransformerEncoder
import torch.multiprocessing
from tqdm import tqdm
from utils import (
    mask_tokens_batch, mask_graph_batch, add_noise_to_3d_structure_batch, to_dense_with_fixed_padding
)
from torch_geometric.data import Batch
from loss import sce_loss, masked_cross_entropy_loss, molecular_denoising_loss
from torch_geometric.utils import scatter
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

# ✅ 只使用单卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_model_dir = 'save_model/'
#BATCH_SIZE = 4  
BATCH_SIZE = 2048
#  EPOCH = 1
EPOCH = 10
LOAD_FROM_LAST = False

from torch_geometric.nn import DataParallel as GeomDataParallel



def get_model_state_dict(model):
    if isinstance(model, GeomDataParallel):
        return model.module.state_dict()
    return model.state_dict()
def save_model(save_tag, epoch, my_model, optimizer, loss):
    saver_dict = {
        'epoch': epoch,
        'model_state_dict': get_model_state_dict(my_model),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': loss
    }

    if save_tag == 'best':
        torch.save(saver_dict, output_model_dir + 'model_best.pth')
    elif save_tag == 'last':
        torch.save(saver_dict, output_model_dir + 'model_last.pth')
    else:
        torch.save(saver_dict, output_model_dir + f'model_{epoch}.pth')


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['best_valid_loss']
    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")
    return epoch, loss


class PreprocessBatch:
    def process(self, batch):
        pos = batch.pos
        pos_mean = global_mean_pool(pos, batch.batch)

        num_tensor = torch.bincount(batch.batch).to(pos.device)

        pos = pos - torch.repeat_interleave(pos_mean, num_tensor, dim=0)
        
        batch.pos = pos



from torch_geometric.data import Batch
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder_1d = transformer_1d()
        self.config = RobertaConfig.from_pretrained('roberta-base')
        self.config.hidden_size = 128  
        self.config.mask_token_id = 2586
        self.config.type_vocab_size=1
        self.config.vocab_size=2586+1
        self.config.max_position_embeddings=60
        self.config.num_attention_heads = 8
        self.decoder_1d = RobertaForMaskedLM(self.config)
        self.encoder_2d = GNN(num_layer=3, hidden_dim=128,output_dim=128)
        self.decoder_2d = GNNDecoder(hidden_dim=128, out_dim=9)
        self.decoder_3d = GNNDecoder(hidden_dim=128, out_dim=3)
        self.encoder_3d = DimeNet(hidden_channels=128,
                                  num_blocks=3,
                                  num_bilinear=8,
                                  num_spherical=7,
                                  num_radial=6,
                                  out_channels=128
                                  )
        self.semmol=SemMol(
            d=128,
            senter_number=50,
            TopK=5,
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
        if isinstance(batch_data, list):
            data = Batch.from_data_list(batch_data)
            batch_size = len(batch_data) 
        else:
            data = batch_data
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1

        # dev = next(self.parameters()).device
        dev = self.token_bias.device
        data = data.to(dev)

        batch_data = data 
        
        batch_size = len(batch_data)
        self.preprocessor.process(batch_data)  

        tokens_emb = torch.tensor(
            np.array(batch_data.tokens),
            dtype=torch.long,
            device=dev
        )
        smi_mask = torch.tensor(
            np.array(batch_data.attention_mask),
            dtype=torch.bool,
            device=dev
        )
        batch_masked_tokens, batch_masked_token_indices=mask_tokens_batch(tokens_emb,smi_mask)

        batch_masked_graphs, batch_masked_graph_indices, batch_masked_atom_indices_2d=mask_graph_batch(batch_data,batch_data.atom2substructure,batch_masked_token_indices)
        batch_masked_graphs = Batch.from_data_list(batch_masked_graphs)

        batch_noisy_positions, batch_noisy_position_indices, batch_masked_atom_indices_3d = add_noise_to_3d_structure_batch(batch_data.atom2substructure,batch_data.pos,batch_data.batch, batch_masked_token_indices, batch_masked_graph_indices)

        masked_token_representation_1d = self.encoder_1d(batch_masked_tokens, smi_mask)  # (batch_size, seq_length,emd_size)。torch.Size([64, 50, 128])

        mask_1d = torch.ones(batch_size, 50, dtype=torch.bool, device=dev)  #torch.Size([64, 50])

        masked_node_representation_2d = self.encoder_2d(batch_masked_graphs.x, batch_masked_graphs.edge_index,
                                                 batch_masked_graphs.edge_attr)  # (num_nodes_in_batch, emb_dim)
        masked_node_representation_2d, mask_2d = to_dense_with_fixed_padding(masked_node_representation_2d,batch_data.batch,50)##torch.Size([64, 50, 128])

        noisy_node_representation_3d = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions,
                                 batch_data.batch) 
        noisy_node_representation_3d, mask_3d = to_dense_with_fixed_padding(noisy_node_representation_3d,
                                                                           batch_data.batch, 50)#torch.Size([64, 50, 128]
        masked_token_representation_1d, masked_node_representation_2d, noisy_node_representation_3d, loss_semmol, info = self.semmol(masked_token_representation_1d, masked_node_representation_2d, noisy_node_representation_3d)

        masked_token_representation_1d = masked_token_representation_1d + self.token_bias.unsqueeze(0).expand(batch_size, -1,-1) 
        masked_node_representation_2d = masked_node_representation_2d + self.graph_bias.unsqueeze(0).expand(batch_size, -1,-1)
        noisy_node_representation_3d = noisy_node_representation_3d + self.molecule_bias.unsqueeze(0).expand(batch_size, -1,-1)

        masked_emd_sum = torch.cat([torch.cat([masked_token_representation_1d, masked_node_representation_2d], dim=1), noisy_node_representation_3d], dim=1)#torch.Size([64, 150, 128])
        mask_label = torch.cat([torch.cat([mask_1d,mask_2d],dim=1),mask_3d],dim=1)

        fussion_feature = self.feature_fussion(masked_emd_sum,mask_label)#torch.Size([64, 150, 128])

        token_representation_1d,node_representation_2d, node_representation_3d = torch.chunk(fussion_feature, 3, dim=1) #torch.Size([64, 50, 128])
        node_representation_2d = node_representation_2d [mask_2d.bool()] #torch.Size([900, 128])
        node_representation_3d = node_representation_3d [mask_3d.bool()] #torch.Size([900, 128])

        predict_token_representation_1d = self.decoder_1d(inputs_embeds=token_representation_1d, attention_mask=smi_mask)
        predict_token_representation_1d = predict_token_representation_1d.logits  #torch.Size([64, 50, 2587])

        predict_node_representation_2d = self.decoder_2d(node_representation_2d,batch_data.edge_index, batch_data.edge_attr)#torch.Size([926, 9])

        predict_node_representation_3d = self.decoder_3d(node_representation_3d, batch_data.edge_index,
                                                         batch_data.edge_attr)   #torch.Size([926, 3])

        return tokens_emb, batch_masked_token_indices, predict_token_representation_1d,\
               batch_data.x[batch_masked_atom_indices_2d],\
               predict_node_representation_2d[batch_masked_atom_indices_2d],\
               batch_data.pos[batch_masked_atom_indices_3d],\
               predict_node_representation_3d[batch_masked_atom_indices_3d],\
               loss_semmol.unsqueeze(0)

def pretrain_train():
    train_data_len = len(train_loader)
    my_model.train()
    total_loss = 0
    use_multi_gpu = isinstance(my_model, GeomDataParallel)
    
    for step, batch_data_list in enumerate(tqdm(train_loader, desc="Training")):

        token_representation_1d, batch_masked_token_indices, predict_token_representation_1d, \
        node_representation_2d, predict_node_representation_2d, \
        node_representation_3d, predict_node_representation_3d, \
        loss_semmol = my_model(batch_data_list)

        loss_semmol = loss_semmol.mean()

        loss_1d = masked_cross_entropy_loss(predict_token_representation_1d, token_representation_1d, batch_masked_token_indices)
        loss_2d = sce_loss(predict_node_representation_2d, node_representation_2d)
        loss_3d = molecular_denoising_loss(predict_node_representation_3d, node_representation_3d)
        
        loss = loss_1d + loss_2d + loss_3d + loss_semmol
        total_loss += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    global optimal_loss
    train_loss = total_loss/train_data_len
    if train_loss<optimal_loss:
        optimal_loss=train_loss
    print('train loss: ', train_loss, ' optim loss: ', optimal_loss)
    return train_loss


def pretrain_evaluate():
    my_model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch_data_list in enumerate(tqdm(valid_loader, desc="Validating")):
            (token_representation_1d, batch_masked_token_indices, predict_token_representation_1d,
             node_representation_2d, predict_node_representation_2d,
             node_representation_3d, predict_node_representation_3d, loss_semmol) = my_model(batch_data_list)
            
            loss_semmol = loss_semmol.mean()

            loss_1d = masked_cross_entropy_loss(predict_token_representation_1d, token_representation_1d,
                                                batch_masked_token_indices)
            loss_2d = sce_loss(predict_node_representation_2d, node_representation_2d)
            loss_3d = molecular_denoising_loss(predict_node_representation_3d, node_representation_3d)
            
            loss = loss_1d + loss_2d + loss_3d + loss_semmol
            
            total_loss += loss.item()

    valid_loss = total_loss / len(valid_loader)
    print('valid loss:', valid_loss)
    return valid_loss


from torch_geometric.loader import DataListLoader, DataLoader
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    dataset = PCQM4Mv2Dataset()
    print('dataset load finish')
    split_idx = dataset.get_idx_split()
    print('split idx load finish')

    randperm = torch.randperm(len(split_idx["train"]))
    # train_idxs = randperm[: int((0.001) * len(split_idx["train"]))]
    
    train_idxs = randperm[:]  
    valid_idxs = randperm[int(0.1 * len(split_idx["train"])):]

    
    if torch.cuda.device_count() > 1:
        TrainLoaderClass = DataListLoader
        ValidLoaderClass = DataListLoader
    else:
        TrainLoaderClass = DataLoader
        ValidLoaderClass = DataLoader


    train_loader = TrainLoaderClass(
        dataset[train_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
        
    )
    valid_loader = ValidLoaderClass(
        dataset[valid_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )


    train_data_len = len(train_loader)
    val_data_len = len(valid_loader)
    print(train_data_len)
    print(val_data_len)

    my_model = MyModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = my_model.to(device)

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            my_model = GeomDataParallel(my_model, device_ids=list(range(torch.cuda.device_count())))
    else:
        print("Single GPU / CPU")

    model_param_group = [{'params': my_model.parameters(), 'lr': 0.0001}]
    optimizer = optim.Adam(model_param_group, weight_decay=1e-5)

    optimal_loss = 1e10

    best_valid_loss = 10000
    current_epoch = 0

    if LOAD_FROM_LAST:
        current_epoch, best_valid_loss = load_checkpoint('save_model/model_last.pth', my_model, optimizer)

    for epoch in range(current_epoch, EPOCH + 1):
        print(f'Epoch: {epoch}')
        print("Training")
        train_loss = pretrain_train()
        print(f'Epoch {epoch + 1}, train loss: {train_loss:.4f}')
        valid_loss = pretrain_evaluate()
        print(f'Epoch {epoch + 1}, valid loss: {valid_loss:.4f}')

        if valid_loss < best_valid_loss:
            save_model('best', epoch + 1, my_model, optimizer, best_valid_loss)
            best_valid_loss = valid_loss
        save_model('temp', epoch + 1, my_model, optimizer, best_valid_loss)
        print("  ")

    save_model('last', EPOCH, my_model, optimizer, best_valid_loss)

