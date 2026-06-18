import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class Chomp(nn.Module):
    """Chomp Layer for Causal Convolution"""
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
class TemEmbed(nn.Module):
    """Temporal Feature Embedding Layer"""
    def __init__(self, D, step_per_day=288):
        super(TemEmbed, self).__init__()
        self.step_per_day = step_per_day
        total_time_dims = 7 + step_per_day
        self.ff = FeedForward([total_time_dims, D, D])
    def forward(self, te):
        """
        te: (B, T, 2)
        return: (B, T, N, D)
        """
        dayofweek = F.one_hot((te[..., 0].long() % 7), 7).float()
        timeofday = F.one_hot((te[..., 1].long() % self.step_per_day), self.step_per_day).float()
        te_concat = torch.cat([dayofweek, timeofday], dim=-1)
        te_concat = te_concat.unsqueeze(dim=2)
        te_emb = self.ff(te_concat)
        return te_emb
class FeedForward(nn.Module):
    """Feedforward Neural Network (MLP)"""
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([
            nn.Linear(fea[i], fea[i + 1])
            for i in range(self.L)
        ])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)
    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x
class SparseSTFExtraction(nn.Module):
    """Spare Spatio-Temporal Feature Extraction"""
    def __init__(self, heads, dims, samples, local_adj):
        super(SparseSTFExtraction, self).__init__()
        features = heads * dims

        self.dyn_spa_mlp = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.ReLU(),
            nn.Linear(features // 2, features),
            nn.Sigmoid()
        )
        self.dyn_tem_mlp = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.ReLU(),
            nn.Linear(features // 2, features),
            nn.Sigmoid()
        )
        self.num_heads = heads
        self.head_dim = dims
        self.samples = samples
        self.local_adj = local_adj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features)
        self.ff = FeedForward([features, features, features], True)
        self.proj = nn.Linear(self.la.shape[1], 1)
    def _add_dynamic_embed(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec):
        context = x[:, -1, :, :].mean(dim=1)  # [B, C]
        dyn_spa_eigval = spa_eigvalue.unsqueeze(0) * (self.dyn_spa_mlp(context) * 2)
        dyn_tem_eigval = tem_eigvalue.unsqueeze(0) * (self.dyn_tem_mlp(context) * 2)
        spa_embed = (spa_eigvec.unsqueeze(0) * dyn_spa_eigval.unsqueeze(1)).unsqueeze(1)
        tem_embed = (tem_eigvec.unsqueeze(0) * dyn_tem_eigval.unsqueeze(1)).unsqueeze(1)
        return x + spa_embed + tem_embed
    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec):
        """
        x: [B,T,N,C]
        return: [B,T,N,C]
        """
        x_ = self._add_dynamic_embedding(x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec)
        B, T, N, C = x_.shape

        Q = self.qfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4).reshape(B * self.h, T, N, self.d)
        K = self.kfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4).reshape(B * self.h, T, N, self.d)
        V = self.vfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4).reshape(B * self.h, T, N, self.d)

        B_h = B * self.h
        K_sample = K[:, :, self.la, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        sampled_nodes = int(self.s * math.log2(N))
        M = self.proj(Q_K_sample).squeeze(-1)
        M_top = M.topk(sampled_nodes, sorted=False)[1]

        Q_reduce = Q[torch.arange(B_h)[:, None, None], torch.arange(T)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) / (self.d ** 0.5)
        attn = torch.softmax(Q_K, dim=-1)

        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        attn_V = torch.matmul(attn, V)
        cp_expanded = cp.expand(-1, -1, -1, self.d)
        value = torch.gather(attn_V, dim=2, index=cp_expanded)
        value = value.view(B, self.h, T, N, self.d).permute(0, 2, 3, 1, 4).reshape(B, T, N, C)
        value = self.ofc(value)
        value = self.ln(value)
        return self.ff(value)
class BiCrossAttn(nn.Module):
    """Bidirectional Cross Attention"""
    def __init__(self, heads, dims):
        super(BiCrossAttn, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        # Low to High
        self.q_l = FeedForward([features, features])
        self.k_h = FeedForward([features, features])
        self.v_h = FeedForward([features, features])
        # High to Low
        self.q_h = FeedForward([features, features])
        self.k_l = FeedForward([features, features])
        self.v_l = FeedForward([features, features])
    def _cross_attn(self, q, k, v, is_mask=True):
        B, T, N, _ = q.shape
        q = q.reshape(B, T, N, self.h, self.d).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d]
        k = k.reshape(B, T, N, self.h, self.d).permute(0, 3, 2, 4, 1)  # [B, h, N, d, T]
        v = v.reshape(B, T, N, self.h, self.d).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d]
        attn = torch.matmul(q, k) / (self.d ** 0.5)
        if is_mask:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, h, N, T, d]
        out = out.permute(0, 3, 2, 1, 4).reshape(B, T, N, -1)
        return out
    def forward(self, xl, xh, te, is_mask=True):
        xl_emb = xl + te
        xh_emb = xh + te
        out_l = self._cross_attn(self.q_l(xl_emb), self.k_h(xh_emb), self.v_h(xh_emb), is_mask)
        out_h = self._cross_attn(self.q_h(xh_emb), self.k_l(xl_emb), self.v_l(xl_emb), is_mask)

        return out_l, out_h
class GateFusion(nn.Module):
    """Gated Fusion"""
    def __init__(self, heads, dims):
        super(GateFusion, self).__init__()
        features = heads * dims
        self.gate = nn.Sequential(
            FeedForward([features * 2, features]),
            nn.Sigmoid()
        )
        self.out_proj = FeedForward([features, features])
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, out_l, out_h, org_l, org_h):
        concat_feature = torch.cat([out_l, out_h], dim=-1)
        z = self.gate(concat_feature)
        fused = z * out_l + (1 - z) * out_h
        fused = self.out_proj(fused) + org_l + org_h
        fused = self.ln(fused)
        return self.ff(fused)
class TemAttn(nn.Module):
    """Temporal Attention"""
    def __init__(self, heads, dims):
        super(TemAttn, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True):
        """
        x: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        """
        x = x + te
        B, T, N, C = x.shape

        query = self.qfc(x).view(B, T, N, self.h, self.d).permute(0, 3, 2, 1, 4).reshape(B * self.h, N, T, self.d)
        key = self.kfc(x).view(B, T, N, self.h, self.d).permute(0, 3, 2, 4, 1).reshape(B * self.h, N, self.d, T)
        value = self.vfc(x).view(B, T, N, self.h, self.d).permute(0, 3, 2, 1, 4).reshape(B * self.h, N, T, self.d)

        attention = torch.matmul(query, key) / (self.d ** 0.5)

        if Mask:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attention = attention.masked_fill(~mask, float('-inf'))

        attention = F.softmax(attention, -1)  # [k*B,N,T,T]

        value = torch.matmul(attention, value).view(B, self.h, N, T, self.d).permute(0, 3, 2, 1, 4).reshape(B, T, N, C)
        value = self.ln(self.ofc(value) + x)
        return self.ff(value)
class TemConv(nn.Module):
    """Temporal Causal Dilated Convolution"""
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=3):
        super(TemConv, self).__init__()
        self.tcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(1, kernel_size),
                          dilation=(1, 2 ** i), padding=(0, (kernel_size - 1) * (2 ** i))),
                Chomp((kernel_size - 1) * (2 ** i)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for i in range(levels)
        ])

    def forward(self, xh):
        xh = xh.permute(0, 3, 2, 1)
        for layer in self.tcn_layers:
            xh = layer(xh)
        xh = xh.permute(0, 3, 2, 1)
        return xh
class DualEncoder(nn.Module):
    def __init__(self, heads, dims, samples, levels, localadj, spawave, temwave):
        super(DualEncoder, self).__init__()
        self.temporal_conv = TemConv(heads * dims, levels=levels)
        self.temporal_att = TemAttn(heads, dims)
        self.spatial_att_l = SparseSTFExtraction(heads, dims, samples, localadj)
        self.spatial_att_h = SparseSTFExtraction(heads, dims, samples, localadj)

        self.spa_eigvalue = nn.Parameter(torch.from_numpy(spawave[0].astype(np.float32)), requires_grad=True)
        self.register_buffer('spa_eigvec', torch.from_numpy(spawave[1].astype(np.float32)))

        self.tem_eigvalue = nn.Parameter(torch.from_numpy(temwave[0].astype(np.float32)), requires_grad=True)
        self.register_buffer('tem_eigvec', torch.from_numpy(temwave[1].astype(np.float32)))

    def forward(self, xl, xh, te):
        """
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        """
        xl = self.temporal_att(xl, te)
        xh = self.temporal_conv(xh)

        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec, self.tem_eigvalue, self.tem_eigvec)
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec, self.tem_eigvalue, self.tem_eigvec)
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        return xl, xh


class AdaptiveFusion(nn.Module):
    def __init__(self, heads, dims):
        super(AdaptiveFusion, self).__init__()
        self.crossAttn = BiCrossAttn(heads, dims)
        self.fusion = GateFusion(heads, dims)
    def forward(self, xl, xh, te, is_mask=True):
        """
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        """
        out_l, out_h = self.crossAttn(xl, xh, te, is_mask)
        return self.fusion(out_l, out_h, xl, xh)
class DSTSANet(nn.Module):
    """Decoupled Spatio-Temporal Sparse Attention Adaptive Network"""
    def __init__(self, heads, dims, layers, samples, levels,local_adj, spa_wave, tem_wave,input_len, output_len):
        super(DSTSANet, self).__init__()
        features = heads * dims
        I = torch.arange(local_adj.shape[0]).unsqueeze(-1)
        local_adj = torch.cat([I, torch.from_numpy(local_adj)], -1)
        self.input_len = input_len

        self.dual_enc = nn.ModuleList([
            DualEncoder(heads, dims, samples, levels, local_adj, spa_wave, tem_wave)
            for _ in range(layers)
        ])
        self.adp_f = AdaptiveFusion(heads, dims)

        self.pre_l = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre_h = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre_fused = nn.Conv2d(input_len, output_len, (1, 1))
        self.trend_proj = nn.Conv2d(input_len, output_len, (1, 1))

        self.start_emb_l = FeedForward([1, features, features])
        self.start_emb_h = FeedForward([1, features, features])

        self.end_emb = FeedForward([features, features, 1])
        self.end_emb_l = FeedForward([features, features, 1])
        self.end_emb_h = FeedForward([features, features, 1])

        self.te_emb = TemEmbed(features)

    def forward(self, xl, xh, te):
        """
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        """
        base_trend = self.trend_proj(xl)
        xl = self.start_emb_l(xl)
        xh = self.start_emb_h(xh)
        te = self.te_emb(te)

        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:, :self.input_len, :, :])

        fused_history = self.adp_f(xl, xh, te[:, :self.input_len, :, :], Mask=True)
        hat_y = self.pre_fused(fused_history)
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        future_te = te[:, self.input_len:, :, :]
        hat_y = hat_y + future_te

        hat_y_out = self.end_emb(hat_y) + base_trend
        hat_y_l_out = self.end_emb_l(hat_y_l)
        hat_y_h_out = self.end_emb_h(hat_y_h)
        return hat_y_out, hat_y_l_out, hat_y_h_out