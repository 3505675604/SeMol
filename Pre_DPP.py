#coding=utf-8
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_geometric\.nn\.data_parallel")

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torch.optim as optim
import torch.multiprocessing
from tqdm import tqdm
import numpy as np

from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader 
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from pcqm4m import PCQM4Mv2Dataset
from model.transformer_model import transformer_1d
from model.SemMol import SemMol
from model.gnn_model import GNN, GNNDecoder
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
from model.feature_fussion import TransformerEncoder
from utils import (
    mask_tokens_batch, mask_graph_batch, add_noise_to_3d_structure_batch, to_dense_with_fixed_padding
)
from loss import sce_loss, masked_cross_entropy_loss, molecular_denoising_loss

torch.multiprocessing.set_sharing_strategy('file_system')

# ====== 训练超参 ======
PER_GPU_BATCH = 256  
EPOCH = 20
LOAD_FROM_LAST = False
output_model_dir = '/data/FL/Semol/save_model/'

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_rank0():
    return get_rank() == 0

def barrier(device):
    if is_dist():
        # dist.barrier()
        dist.barrier(device_ids=[device.index])

def setup_ddp():
    # torchrun 会自动提供：RANK, WORLD_SIZE, LOCAL_RANK
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()

def get_model_state_dict(model):
    # DDP 包裹后，真实模型在 model.module
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()

def save_model(save_tag, epoch, my_model, optimizer, loss):
    if not is_rank0():
        return

    os.makedirs(output_model_dir, exist_ok=True)
    saver_dict = {
        'epoch': epoch,
        'model_state_dict': get_model_state_dict(my_model),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': loss
    }

    if save_tag == 'best':
        torch.save(saver_dict, os.path.join(output_model_dir, 'model_best.pth'))
    elif save_tag == 'last':
        torch.save(saver_dict, os.path.join(output_model_dir, 'model_last.pth'))
    else:
        torch.save(saver_dict, os.path.join(output_model_dir, f'model_{epoch}.pth'))

def load_checkpoint(filename, model, optimizer, map_location):
    ckpt = torch.load(filename, map_location=map_location)
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    loss = ckpt['best_valid_loss']
    if is_rank0():
        print(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")
    return epoch, loss

class PreprocessBatch:
    def process(self, batch):
        pos = batch.pos
        pos_mean = global_mean_pool(pos, batch.batch)
        num_tensor = torch.bincount(batch.batch).to(pos.device)
        pos = pos - torch.repeat_interleave(pos_mean, num_tensor, dim=0)
        batch.pos = pos

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder_1d = transformer_1d()
        self.config = RobertaConfig.from_pretrained('roberta-base')
        self.config.hidden_size = 128
        self.config.mask_token_id = 2586
        self.config.type_vocab_size = 1
        self.config.vocab_size = 2586 + 1
        self.config.max_position_embeddings = 60
        self.config.num_attention_heads = 8
        self.decoder_1d = RobertaForMaskedLM(self.config)

        self.encoder_2d = GNN(num_layer=3, hidden_dim=128, output_dim=128)
        self.decoder_2d = GNNDecoder(hidden_dim=128, out_dim=9)
        self.decoder_3d = GNNDecoder(hidden_dim=128, out_dim=3)
        self.encoder_3d = DimeNet(hidden_channels=128,
                                  num_blocks=3,
                                  num_bilinear=8,
                                  num_spherical=7,
                                  num_radial=6,
                                  out_channels=128)

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
        # DDP + PyG DataLoader 返回的是 Batch（不是 list）
        if isinstance(batch_data, list):
            data = Batch.from_data_list(batch_data)
            batch_size = len(batch_data)
        else:
            data = batch_data
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1

        dev = self.token_bias.device
        data = data.to(dev)

        batch_data = data
        self.preprocessor.process(batch_data)

        tokens_emb = torch.tensor(np.array(batch_data.tokens), dtype=torch.long, device=dev)
        smi_mask = torch.tensor(np.array(batch_data.attention_mask), dtype=torch.bool, device=dev)

        batch_masked_tokens, batch_masked_token_indices = mask_tokens_batch(tokens_emb, smi_mask)
        batch_masked_graphs, batch_masked_graph_indices, batch_masked_atom_indices_2d = mask_graph_batch(
            batch_data, batch_data.atom2substructure, batch_masked_token_indices
        )
        batch_masked_graphs = Batch.from_data_list(batch_masked_graphs)

        batch_noisy_positions, batch_noisy_position_indices, batch_masked_atom_indices_3d = add_noise_to_3d_structure_batch(
            batch_data.atom2substructure, batch_data.pos, batch_data.batch,
            batch_masked_token_indices, batch_masked_graph_indices
        )

        masked_token_representation_1d = self.encoder_1d(batch_masked_tokens, smi_mask)
        mask_1d = torch.ones(batch_size, 50, dtype=torch.bool, device=dev)

        masked_node_representation_2d = self.encoder_2d(
            batch_masked_graphs.x, batch_masked_graphs.edge_index, batch_masked_graphs.edge_attr
        )
        masked_node_representation_2d, mask_2d = to_dense_with_fixed_padding(
            masked_node_representation_2d, batch_data.batch, 50
        )

        noisy_node_representation_3d = self.encoder_3d(
            batch_data.x[:, 0].long(), batch_noisy_positions, batch_data.batch
        )
        noisy_node_representation_3d, mask_3d = to_dense_with_fixed_padding(
            noisy_node_representation_3d, batch_data.batch, 50
        )

        masked_token_representation_1d, masked_node_representation_2d, noisy_node_representation_3d, loss_semmol, info = \
            self.semmol(masked_token_representation_1d, masked_node_representation_2d, noisy_node_representation_3d)

        masked_token_representation_1d = masked_token_representation_1d + self.token_bias.unsqueeze(0).expand(batch_size, -1, -1)
        masked_node_representation_2d = masked_node_representation_2d + self.graph_bias.unsqueeze(0).expand(batch_size, -1, -1)
        noisy_node_representation_3d = noisy_node_representation_3d + self.molecule_bias.unsqueeze(0).expand(batch_size, -1, -1)

        masked_emd_sum = torch.cat([
            torch.cat([masked_token_representation_1d, masked_node_representation_2d], dim=1),
            noisy_node_representation_3d
        ], dim=1)
        mask_label = torch.cat([torch.cat([mask_1d, mask_2d], dim=1), mask_3d], dim=1)

        fussion_feature = self.feature_fussion(masked_emd_sum, mask_label)

        token_representation_1d, node_representation_2d, node_representation_3d = torch.chunk(fussion_feature, 3, dim=1)
        node_representation_2d = node_representation_2d[mask_2d.bool()]
        node_representation_3d = node_representation_3d[mask_3d.bool()]

        predict_token_representation_1d = self.decoder_1d(inputs_embeds=token_representation_1d, attention_mask=smi_mask).logits
        predict_node_representation_2d = self.decoder_2d(node_representation_2d, batch_data.edge_index, batch_data.edge_attr)
        predict_node_representation_3d = self.decoder_3d(node_representation_3d, batch_data.edge_index, batch_data.edge_attr)

        return (tokens_emb, batch_masked_token_indices, predict_token_representation_1d,
                batch_data.x[batch_masked_atom_indices_2d],
                predict_node_representation_2d[batch_masked_atom_indices_2d],
                batch_data.pos[batch_masked_atom_indices_3d],
                predict_node_representation_3d[batch_masked_atom_indices_3d],
                loss_semmol.unsqueeze(0))

def train_one_epoch(model, optimizer, train_loader):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training(rank={get_rank()})", disable=not is_rank0()):
        (token_representation_1d, batch_masked_token_indices, predict_token_representation_1d,
         node_representation_2d, predict_node_representation_2d,
         node_representation_3d, predict_node_representation_3d, loss_semmol) = model(batch)

        loss_semmol = loss_semmol.mean()
        loss_1d = masked_cross_entropy_loss(predict_token_representation_1d, token_representation_1d, batch_masked_token_indices)
        loss_2d = sce_loss(predict_node_representation_2d, node_representation_2d)
        loss_3d = molecular_denoising_loss(predict_node_representation_3d, node_representation_3d)
        loss = loss_1d + loss_2d + loss_3d + loss_semmol

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())

    # return total_loss / max(1, len(train_loader))

    avg = total_loss
    cnt = len(train_loader)
    if is_dist():
        t = torch.tensor([avg, cnt], device=next(model.parameters()).device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        avg = (t[0] / t[1]).item()
    else:
        avg = avg / max(1, cnt)
    return avg


@torch.no_grad()
def evaluate(model, valid_loader):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(valid_loader, desc=f"Valid(rank={get_rank()})", disable=not is_rank0()):
        (token_representation_1d, batch_masked_token_indices, predict_token_representation_1d,
         node_representation_2d, predict_node_representation_2d,
         node_representation_3d, predict_node_representation_3d, loss_semmol) = model(batch)

        loss_semmol = loss_semmol.mean()
        loss_1d = masked_cross_entropy_loss(predict_token_representation_1d, token_representation_1d, batch_masked_token_indices)
        loss_2d = sce_loss(predict_node_representation_2d, node_representation_2d)
        loss_3d = molecular_denoising_loss(predict_node_representation_3d, node_representation_3d)
        loss = loss_1d + loss_2d + loss_3d + loss_semmol
        total_loss += float(loss.item())

    # return total_loss / max(1, len(valid_loader))
    avg = total_loss
    cnt = len(valid_loader)
    if is_dist():
        t = torch.tensor([avg, cnt], device=next(model.parameters()).device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        avg = (t[0] / t[1]).item()
    else:
        avg = avg / max(1, cnt)
    return avg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    set_seed(42) 

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    dataset = PCQM4Mv2Dataset()
    if is_rank0():
        print('dataset load finish')
    split_idx = dataset.get_idx_split()
    if is_rank0():
        print('split idx load finish')

    # randperm = torch.randperm(len(split_idx["train"]))
    # train_idxs = randperm[:]
    # valid_idxs = randperm[int(0.1 * len(split_idx["train"])):]

    randperm = torch.randperm(len(split_idx["train"])).tolist()
    num_train = len(split_idx["train"])
    split_point = int(num_train * 0.9)
    train_idxs = randperm[:split_point]
    valid_idxs = randperm[split_point:]

    train_set = Subset(dataset, train_idxs)
    valid_set = Subset(dataset, valid_idxs)

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        train_set,
        batch_size=PER_GPU_BATCH,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=PER_GPU_BATCH,
        sampler=valid_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    model = MyModel().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-4}], weight_decay=1e-5)

    best_valid_loss = 1e10
    current_epoch = 0

    if LOAD_FROM_LAST and os.path.exists(os.path.join(output_model_dir, 'model_last.pth')):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        current_epoch, best_valid_loss = load_checkpoint(os.path.join(output_model_dir, 'model_last.pth'), model.module, optimizer, map_location)

    for epoch in range(current_epoch, EPOCH):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        if is_rank0():
            print(f"\nEpoch: {epoch+1}/{EPOCH}")

        train_loss = train_one_epoch(model, optimizer, train_loader)
        valid_loss = evaluate(model, valid_loader)

        if is_rank0():
            print(f"train loss: {train_loss:.6f} | valid loss: {valid_loss:.6f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_model('best', epoch + 1, model, optimizer, best_valid_loss)

            save_model('temp', epoch + 1, model, optimizer, best_valid_loss)

        barrier(device)
    barrier(device) 
    if is_rank0():
        save_model('last', EPOCH, model, optimizer, best_valid_loss)
    barrier(device) 
    cleanup_ddp()

if __name__ == "__main__":
    main()
