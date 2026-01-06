import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


class SemMol(nn.Module):
    def __init__(
        self,
        d=128,
        K=100,
        M=5,
        beta=0.9,
        tau=0.07,
        T=0.5,
        lambda_2d=1.0,
        lambda_3d=1.0,
        kmeans_batch_size=1024,
        kmeans_random_state=42,
        loss_reduction="mean",
        use_fusion=True,
        use_output_norm=True,
        warmup_steps=0,
        fine_tune=False,

        senter_number=None,   
        TopK=None,           
        lambda_1d=1.0,        
        anchor_mode="1d",     
        use_bidirectional=False,  
        use_cross_attention=False, 
        adaptive_tau=False,       
        **kwargs,                  
    ):
        super().__init__()
        assert loss_reduction in ["sum", "mean"]
        if senter_number is not None:
            K = senter_number
        if TopK is not None:
            M = TopK

        self.d = d
        self.K = K
        self.M = M
        self.beta = beta
        self.tau = tau
        self.T = T
        self.lambda_2d = lambda_2d
        self.lambda_3d = lambda_3d

        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_random_state = kmeans_random_state

        self.loss_reduction = loss_reduction
        self.use_fusion = use_fusion
        self.use_output_norm = use_output_norm
        self.warmup_steps = warmup_steps
        self.fine_tune = fine_tune

        self.MLP = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        self.W_b = nn.Linear(d, d, bias=True)
        self.W_c = nn.Linear(d, d, bias=True)

        self.register_buffer("C_2D", torch.zeros(K, d))
        self.register_buffer("C_3D", torch.zeros(K, d))
        self.centers_initialized = False

        if use_fusion:
            self.fusion_1d = nn.Sequential(
                nn.Linear(d * 2, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
            self.fusion_2d = nn.Sequential(
                nn.Linear(d * 2, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
            self.fusion_3d = nn.Sequential(
                nn.Linear(d * 2, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )

        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def project(self, z_1d, z_2d, z_3d):
        h_1D = self.MLP(z_1d)              
        h_2D = self.MLP(self.W_b(z_2d))    
        h_3D = self.MLP(self.W_c(z_3d))   
        return h_1D, h_2D, h_3D

    def pool_molecule(self, h_seq):
        return h_seq.mean(dim=1)  

    def initialize_centers(self, h2_mol, h3_mol):
        device = h2_mol.device

        def kmeans_or_fallback(x, K):
            if x.size(0) >= K:
                km = MiniBatchKMeans(
                    n_clusters=K,
                    random_state=self.kmeans_random_state,
                    batch_size=self.kmeans_batch_size,
                )
                km.fit(x.detach().cpu().numpy())
                return torch.from_numpy(km.cluster_centers_).float()
            idx = torch.randint(0, x.size(0), (K,), device=x.device)
            return x[idx].detach().cpu().float()

        self.C_2D = kmeans_or_fallback(h2_mol, self.K).to(device)
        self.C_3D = kmeans_or_fallback(h3_mol, self.K).to(device)
        self.centers_initialized = True

    @torch.no_grad()
    def update_centers(self, h2_mol, h3_mol):
        def ema_update(h_mol, C_m):
            dist = torch.cdist(h_mol, C_m, p=2)  
            assign = dist.argmin(dim=1)
            for k in range(self.K):
                S_k = (assign == k)
                if S_k.any():
                    mean_Sk = h_mol[S_k].mean(dim=0)
                    C_m[k] = self.beta * C_m[k] + (1.0 - self.beta) * mean_Sk

        ema_update(h2_mol, self.C_2D)
        ema_update(h3_mol, self.C_3D)

    def topM_by_cosine(self, h_anchor, centers):
        M = min(self.M, centers.size(0))
        a = F.normalize(h_anchor, p=2, dim=1)     
        c = F.normalize(centers, p=2, dim=1)      
        sim = a @ c.t()                          
        I_i = torch.topk(sim, k=M, dim=1, sorted=True).indices
        sim_top = torch.gather(sim, dim=1, index=I_i)    
        mu_top = centers[I_i]                             
        return I_i, sim_top, mu_top

    def build_positive_final(self, h_anchor, centers_2d, centers_3d):
        I_2D, sim_2D, mu_top_2D = self.topM_by_cosine(h_anchor, centers_2d)
        a_2D = torch.softmax(sim_2D / self.tau, dim=1)                
        h_pos_2D = torch.sum(a_2D.unsqueeze(2) * mu_top_2D, dim=1)

        I_3D, sim_3D, mu_top_3D = self.topM_by_cosine(h_anchor, centers_3d)
        a_3D = torch.softmax(sim_3D / self.tau, dim=1)                 
        h_pos_3D = torch.sum(a_3D.unsqueeze(2) * mu_top_3D, dim=1)

        h_pos_final = h_pos_2D + h_pos_3D                              
        return h_pos_final, (I_2D, I_3D)

    def build_negatives_final(self, h_anchor, I_i, centers):

        B = h_anchor.size(0)
        device = h_anchor.device
        K = centers.size(0)
        all_idx = torch.arange(K, device=device)

        a = F.normalize(h_anchor, p=2, dim=1)
        c_norm = F.normalize(centers, p=2, dim=1)

        neg_list = []
        for i in range(B):
            mask = torch.ones(K, dtype=torch.bool, device=device)
            mask[I_i[i]] = False                         
            I_bar = all_idx[mask]

            mu_neg = centers[I_bar]                       
            mu_neg_norm = c_norm[I_bar]
            cos_sim = (a[i].unsqueeze(0) @ mu_neg_norm.t()).squeeze(0)
            keep = cos_sim > self.T                         
            neg_list.append(mu_neg[keep])
        return neg_list

    def infonce_sum(self, h_anchor, h_pos_final, neg_list):
        B = h_anchor.size(0)
        a = F.normalize(h_anchor, p=2, dim=1)
        p = F.normalize(h_pos_final, p=2, dim=1)

        losses = []
        for i in range(B):
            neg = neg_list[i]
            if neg.numel() == 0:
                losses.append(torch.zeros([], device=h_anchor.device, dtype=h_anchor.dtype))
                continue

            neg_n = F.normalize(neg, p=2, dim=1)
            pos_logit = torch.sum(a[i] * p[i]) / self.tau
            neg_logits = (neg_n @ a[i]) / self.tau
            logits = torch.cat([pos_logit.unsqueeze(0), neg_logits], dim=0)       
            loss_i = -F.log_softmax(logits, dim=0)[0]       
            losses.append(loss_i)

        per_i = torch.stack(losses, dim=0)                 
        return per_i.sum(), per_i.mean()

    def forward(self, z_1d, z_2d, z_3d, update_centers=True):

        if self.training:
            self._step += 1

        B, L, _ = z_1d.shape

        h_1D, h_2D, h_3D = self.project(z_1d, z_2d, z_3d)

        h_anchor = self.pool_molecule(h_1D)   
        h2_mol = self.pool_molecule(h_2D)
        h3_mol = self.pool_molecule(h_3D)

        if not self.centers_initialized:
            self.initialize_centers(h2_mol, h3_mol)

       
        do_update = (
            update_centers
            and self.training
            and (int(self._step.item()) > self.warmup_steps)
        )
        if do_update:
            self.update_centers(h2_mol, h3_mol)

        
        h_pos_final, (I_2D, I_3D) = self.build_positive_final(h_anchor, self.C_2D, self.C_3D)

        
        neg_2D = self.build_negatives_final(h_anchor, I_2D, self.C_2D)
        neg_3D = self.build_negatives_final(h_anchor, I_3D, self.C_3D)

        
        sum_L_2d, mean_L_2d = self.infonce_sum(h_anchor, h_pos_final, neg_2D)
        sum_L_3d, mean_L_3d = self.infonce_sum(h_anchor, h_pos_final, neg_3D)

        
        L_total_sum = self.lambda_2d * sum_L_2d + self.lambda_3d * sum_L_3d

        
        if self.loss_reduction == "sum":
            loss_total = L_total_sum
        else:
            
            loss_total = self.lambda_2d * mean_L_2d + self.lambda_3d * mean_L_3d

        
        if self.use_fusion:
            pos_seq = h_pos_final.unsqueeze(1).expand(-1, L, -1)  
            h1_out = self.fusion_1d(torch.cat([h_1D, pos_seq], dim=2))
            h2_out = self.fusion_2d(torch.cat([h_2D, pos_seq], dim=2))
            h3_out = self.fusion_3d(torch.cat([h_3D, pos_seq], dim=2))
        else:
            h1_out, h2_out, h3_out = h_1D, h_2D, h_3D

        if self.use_output_norm:
            h1_out = F.normalize(h1_out, p=2, dim=2)
            h2_out = F.normalize(h2_out, p=2, dim=2)
            h3_out = F.normalize(h3_out, p=2, dim=2)

        if self.fine_tune:
            combined_output = torch.cat([h1_out, h2_out, h3_out], dim=2)  
            return combined_output, loss_total

        info = {
            "step": int(self._step.item()),
            "warmup_steps": self.warmup_steps,
            "centers_updated": bool(do_update),
            "tau": float(self.tau),
            "T": float(self.T),
            "lambda_2d": float(self.lambda_2d),
            "lambda_3d": float(self.lambda_3d),
            "sum_L_2d": float(sum_L_2d.detach().cpu()),
            "sum_L_3d": float(sum_L_3d.detach().cpu()),
            "mean_L_2d": float(mean_L_2d.detach().cpu()),
            "mean_L_3d": float(mean_L_3d.detach().cpu()),
            "loss_total": float(loss_total.detach().cpu()),
        }

        return h1_out, h2_out, h3_out, loss_total, info
