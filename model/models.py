import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemEmbedding(nn.Module):
    def __init__(self, D, step_per_day=288):
        super(TemEmbedding, self).__init__()
        self.step_per_day = step_per_day
        total_time_dims = 7 + step_per_day
        self.ff_te = FeedForward([total_time_dims, D, D])

    def forward(self, te):
        """
        te: (B, T, 2)
        return: (B, T, N, D)
        """
        dayofweek = F.one_hot((te[..., 0].long() % 7), 7).float()
        timeofday = F.one_hot((te[..., 1].long() % self.step_per_day), self.step_per_day).float()
        te_concat = torch.cat([dayofweek, timeofday], dim=-1)
        # (B, T, D)
        te_concat = te_concat.unsqueeze(dim=2)
        # (B, T, 1, D)
        te_emb = self.ff_te(te_concat)
        return te_emb


class FeedForward(nn.Module):
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


# Ablation 2
class SparseSTFExtraction(nn.Module):
    def __init__(self, heads, dims, samples, local_adj, graph_mode='both'):
        """
        graph_mode: 'both' - use both spatial and temporal graph (default)
                    'spatial_only' - only use spatial graph
                    'temporal_only' - only use temporal graph
        """
        super(SparseSTFExtraction, self).__init__()
        self.graph_mode = graph_mode
        features = heads * dims
        if graph_mode != 'temporal_only':
            self.dyn_spa_mlp = nn.Sequential(
                nn.Linear(features, features // 2),
                nn.ReLU(),
                nn.Linear(features // 2, features),
                nn.Sigmoid()
            )
        if graph_mode != 'spatial_only':
            self.dyn_tem_mlp = nn.Sequential(
                nn.Linear(features, features // 2),
                nn.ReLU(),
                nn.Linear(features // 2, features),
                nn.Sigmoid()
            )
        self.h = heads
        self.d = dims
        self.s = samples
        self.la = local_adj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features)
        self.ff = FeedForward([features, features, features], True)
        self.proj = nn.Linear(self.la.shape[1], 1)

    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec):
        """
        x: [B,T,N,C]
        return: [B,T,N,C]
        """
        context = x[:, -1, :, :].mean(dim=1)

        if self.graph_mode == 'spatial_only':
            dyn_spa_eigval = spa_eigvalue.unsqueeze(0) * (self.dyn_spa_mlp(context) * 2)
            spa_embed = (spa_eigvec.unsqueeze(0) * dyn_spa_eigval.unsqueeze(1)).unsqueeze(1)
            x_ = x + spa_embed
        elif self.graph_mode == 'temporal_only':
            dyn_tem_eigval = tem_eigvalue.unsqueeze(0) * (self.dyn_tem_mlp(context) * 2)
            tem_embed = (tem_eigvec.unsqueeze(0) * dyn_tem_eigval.unsqueeze(1)).unsqueeze(1)
            x_ = x + tem_embed
        else:
            dyn_spa_eigval = spa_eigvalue.unsqueeze(0) * (self.dyn_spa_mlp(context) * 2)
            dyn_tem_eigval = tem_eigvalue.unsqueeze(0) * (self.dyn_tem_mlp(context) * 2)
            spa_embed = (spa_eigvec.unsqueeze(0) * dyn_spa_eigval.unsqueeze(1)).unsqueeze(1)
            tem_embed = (tem_eigvec.unsqueeze(0) * dyn_tem_eigval.unsqueeze(1)).unsqueeze(1)
            x_ = x + spa_embed + tem_embed
        B, T, N, C = x_.shape

        Q = self.qfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4).reshape(B * self.h, T, N, self.d)
        K = self.kfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4).reshape(B * self.h, T, N, self.d)
        V = self.vfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4).reshape(B * self.h, T, N, self.d)

        B_h = B * self.h
        K_sample = K[:, :, self.la, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        Sampled_Nodes = int(self.s * math.log2(N))
        M = self.proj(Q_K_sample).squeeze(-1)
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]

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

    @torch.no_grad()
    def get_target_attention(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec, target_node):
        """
        x: [B, T, N, C]
        return:
            attn_target: [B, H, T, N]
        """
        context = x[:, -1, :, :].mean(dim=1)  # [B, C]

        if self.graph_mode == 'spatial_only':
            dyn_spa_eigval = spa_eigvalue.unsqueeze(0) * (self.dyn_spa_mlp(context) * 2)
            spa_embed = (spa_eigvec.unsqueeze(0) * dyn_spa_eigval.unsqueeze(1)).unsqueeze(1)
            x_ = x + spa_embed
        elif self.graph_mode == 'temporal_only':
            dyn_tem_eigval = tem_eigvalue.unsqueeze(0) * (self.dyn_tem_mlp(context) * 2)
            tem_embed = (tem_eigvec.unsqueeze(0) * dyn_tem_eigval.unsqueeze(1)).unsqueeze(1)
            x_ = x + tem_embed
        else:
            dyn_spa_eigval = spa_eigvalue.unsqueeze(0) * (self.dyn_spa_mlp(context) * 2)
            dyn_tem_eigval = tem_eigvalue.unsqueeze(0) * (self.dyn_tem_mlp(context) * 2)
            spa_embed = (spa_eigvec.unsqueeze(0) * dyn_spa_eigval.unsqueeze(1)).unsqueeze(1)
            tem_embed = (tem_eigvec.unsqueeze(0) * dyn_tem_eigval.unsqueeze(1)).unsqueeze(1)
            x_ = x + spa_embed + tem_embed
        B, T, N, C = x_.shape

        Q = self.qfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4)  # [B,H,T,N,d]
        K = self.kfc(x_).view(B, T, N, self.h, self.d).permute(0, 3, 1, 2, 4)  # [B,H,T,N,d]

        q_target = Q[:, :, :, target_node, :]  # [B,H,T,d]
        scores = torch.einsum('bhtd,bhtnd->bhtn', q_target, K) / (self.d ** 0.5)
        attn_target = torch.softmax(scores, dim=-1)  # [B,H,T,N]

        return attn_target


# Ablation 3
class BidirectCrossFreqInteraction(nn.Module):
    def __init__(self, heads, dims):
        super(BidirectCrossFreqInteraction, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        # L2H
        self.q_l = FeedForward([features, features])
        self.k_h = FeedForward([features, features])
        self.v_h = FeedForward([features, features])
        # H2L
        self.q_h = FeedForward([features, features])
        self.k_l = FeedForward([features, features])
        self.v_l = FeedForward([features, features])

    def _cross_attn(self, q, k, v, Mask=True):
        B, T, N, _ = q.shape
        q = q.reshape(B, T, N, self.h, self.d).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d]
        k = k.reshape(B, T, N, self.h, self.d).permute(0, 3, 2, 4, 1)  # [B, h, N, d, T]
        v = v.reshape(B, T, N, self.h, self.d).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d]
        attn = torch.matmul(q, k) / (self.d ** 0.5)
        if Mask:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, h, N, T, d]
        out = out.permute(0, 3, 2, 1, 4).reshape(B, T, N, -1)
        return out

    def forward(self, xl, xh, te, Mask=True):
        xl_emb = xl + te
        xh_emb = xh + te
        out_l = self._cross_attn(self.q_l(xl_emb), self.k_h(xh_emb), self.v_h(xh_emb), Mask=Mask)
        out_h = self._cross_attn(self.q_h(xh_emb), self.k_l(xl_emb), self.v_l(xl_emb), Mask=Mask)

        return out_l, out_h


# Ablation 4
class GateAggregation(nn.Module):
    def __init__(self, heads, dims):
        super(GateAggregation, self).__init__()
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
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=3):
        super(TemConv, self).__init__()
        self.tcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(1, kernel_size),
                          dilation=(1, 2 ** i), padding=(0, (kernel_size - 1) * (2 ** i))),
                Chomp1d((kernel_size - 1) * (2 ** i)),
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


# Ablation 1
class FSTEncoder(nn.Module):
    def __init__(self, heads, dims, samples, levels, localadj, spawave, temwave, wo_sparse_spatial=False,
                 graph_mode='both'):
        super(FSTEncoder, self).__init__()
        self.wo_sparse_spatial = wo_sparse_spatial

        self.temporal_conv = TemConv(heads * dims, levels=levels)
        self.temporal_att = TemAttn(heads, dims)

        self.spatial_att_l = SparseSTFExtraction(heads, dims, samples, localadj, graph_mode=graph_mode)
        self.spatial_att_h = SparseSTFExtraction(heads, dims, samples, localadj, graph_mode=graph_mode)

        spa_eigvalue = torch.from_numpy(spawave[0].astype(np.float32))
        self.spa_eigvalue = nn.Parameter(spa_eigvalue, requires_grad=True)
        self.register_buffer('spa_eigvec', torch.from_numpy(spawave[1].astype(np.float32)))

        tem_eigvalue = torch.from_numpy(temwave[0].astype(np.float32))
        self.tem_eigvalue = nn.Parameter(tem_eigvalue, requires_grad=True)
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
        if not self.wo_sparse_spatial:
            spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec, self.tem_eigvalue, self.tem_eigvec)
            spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec, self.tem_eigvalue, self.tem_eigvec)
            xl = spa_statesl + xl
            xh = spa_statesh + xh
        return xl, xh


class CrossFreqFusion(nn.Module):
    def __init__(self, heads, dims, wo_bidirectional_cross=False, wo_gated_fusion=False):
        super(CrossFreqFusion, self).__init__()
        self.wo_bidirectional_cross = wo_bidirectional_cross
        self.wo_gated_fusion = wo_gated_fusion
        self.interaction = BidirectCrossFreqInteraction(heads, dims)
        self.fusion = GateAggregation(heads, dims)

    def forward(self, xl, xh, te, Mask=True):
        """
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        """
        if self.wo_bidirectional_cross:
            out_l, out_h = xl, xh
        else:
            out_l, out_h = self.interaction(xl, xh, te, Mask=Mask)
        if self.wo_gated_fusion:
            return out_l + out_h
        else:
            return self.fusion(out_l, out_h, xl, xh)


class DSTSANet(nn.Module):
    def __init__(self, heads, dims, layers, samples, levels,
                 local_adj, spa_wave, tem_wave,
                 input_len, output_len, ablation=None, encoder_type='dual'):
        super(DSTSANet, self).__init__()
        features = heads * dims
        I = torch.arange(local_adj.shape[0]).unsqueeze(-1)
        local_adj = torch.cat([I, torch.from_numpy(local_adj)], -1)
        self.input_len = input_len
        if ablation is None:
            ablation = {}
        self.ablation = {
            "wo_dual_frequency": ablation.get("wo_dual_frequency", False),
            "wo_sparse_spatial": ablation.get("wo_sparse_spatial", False),
            "wo_bidirectional_cross": ablation.get("wo_bidirectional_cross", False),
            "wo_gated_fusion": ablation.get("wo_gated_fusion", False),
            "wo_temporal_graph": ablation.get("wo_temporal_graph", False),
            "wo_spatial_graph": ablation.get("wo_spatial_graph", False),
        }
        self.encoder_type = encoder_type
        # Determine graph_mode for SparseSTFExtraction
        graph_mode = 'both'
        if self.ablation['wo_temporal_graph']:
            graph_mode = 'spatial_only'
        elif self.ablation['wo_spatial_graph']:
            graph_mode = 'temporal_only'
        # Lazy import to avoid circular import with model.ablation
        from model.ablation import ABLATION_ENCODERS
        if encoder_type == 'dual':
            encoder_cls = FSTEncoder
        elif encoder_type in ABLATION_ENCODERS:
            encoder_cls = ABLATION_ENCODERS[encoder_type]
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        # Pass through any ablation kwargs the encoder might honor.
        enc_kwargs = {
            'wo_sparse_spatial': self.ablation['wo_sparse_spatial'],
            'graph_mode': graph_mode,
        }
        # Filter to kwargs the chosen encoder actually accepts.
        import inspect
        sig = inspect.signature(encoder_cls.__init__)
        accepted = {k: v for k, v in enc_kwargs.items() if k in sig.parameters}
        self.dual_enc = nn.ModuleList([
            encoder_cls(heads, dims, samples, levels, local_adj, spa_wave, tem_wave, **accepted)
            for _ in range(layers)
        ])
        self.adp_f = CrossFreqFusion(heads, dims,
                                    wo_bidirectional_cross=self.ablation["wo_bidirectional_cross"],
                                    wo_gated_fusion=self.ablation["wo_gated_fusion"])

        self.pre_l = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre_h = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre_fused = nn.Conv2d(input_len, output_len, (1, 1))
        self.trend_proj = nn.Conv2d(input_len, output_len, (1, 1))

        self.start_emb_l = FeedForward([1, features, features])
        self.start_emb_h = FeedForward([1, features, features])

        self.end_emb = FeedForward([features, features, 1])
        self.end_emb_l = FeedForward([features, features, 1])
        self.end_emb_h = FeedForward([features, features, 1])

        self.te_emb = TemEmbedding(features)

    def forward(self, xl, xh, te):
        """
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        """
        base_trend = self.trend_proj(xl)
        xl = self.start_emb_l(xl)
        if self.ablation["wo_dual_frequency"]:
            xh = xl
        else:
            xh = self.start_emb_h(xh)
        te = self.te_emb(te)

        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:, :self.input_len, :, :])
        if self.ablation["wo_dual_frequency"]:
            fused_history = xl
            hat_y = self.pre_l(fused_history)
            hat_y_l = hat_y
            hat_y_h = hat_y
            future_te = te[:, self.input_len:, :, :]
            hat_y = hat_y + future_te
        else:
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

    def get_embeddings(self, xl, xh, te):
        """
        Extract intermediate embeddings for visualization.
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: fused embedding [B,T,N,F], xl embedding [B,T,N,F], xh embedding [B,T,N,F]
        """
        xl = self.start_emb_l(xl)
        if self.ablation["wo_dual_frequency"]:
            xh = xl
        else:
            xh = self.start_emb_h(xh)
        te_emb = self.te_emb(te)

        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te_emb[:, :self.input_len, :, :])
        if self.ablation["wo_dual_frequency"]:
            fused = xl
        else:
            fused = self.adp_f(xl, xh, te_emb[:, :self.input_len, :, :], Mask=True)

        return fused, xl, xh

    @torch.no_grad()
    def predict_and_get_case_data(self, xl, xh, te, target_node=62):
        """
        专门用于论文 Case 分析的接口。
        参数:
            xl: [B, T, N, F] 低频输入
            xh: [B, T, N, F] 高频输入
            te: [B, T+12, N, 2] 时间嵌入 (包含历史和未来12个步长)
            target_node: int, 目标节点索引 (默认 62)
        返回:
            hat_y_out: [B, 12, N, 1] 模型预测的 12 小时结果
            attn_weights: [B, H, T, N] 目标节点的空间注意力权重（用于绘制局部图）
        """
        base_trend = self.trend_proj(xl)
        xl_emb = self.start_emb_l(xl)
        if self.ablation["wo_dual_frequency"]:
            xh_emb = xl_emb
        else:
            xh_emb = self.start_emb_h(xh)

        te_emb = self.te_emb(te)

        attn_weights_list = []

        # 逐层通过 FSTEncoder，并提取目标节点的注意力
        for enc in self.dual_enc:
            if not self.ablation.get('wo_sparse_spatial', False):
                # 提取低频分支的空间注意力作为局部图的依据
                attn = enc.spatial_att_l.get_target_attention(
                    xl_emb, enc.spa_eigvalue, enc.spa_eigvec,
                    enc.tem_eigvalue, enc.tem_eigvec, target_node
                )
                attn_weights_list.append(attn)

            xl_emb, xh_emb = enc(xl_emb, xh_emb, te_emb[:, :self.input_len, :, :])

        # 融合与预测
        if self.ablation["wo_dual_frequency"]:
            fused_history = xl_emb
            hat_y = self.pre_l(fused_history)
            future_te = te_emb[:, self.input_len:, :, :]
            hat_y = hat_y + future_te
        else:
            fused_history = self.adp_f(xl_emb, xh_emb, te_emb[:, :self.input_len, :, :], Mask=True)
            hat_y = self.pre_fused(fused_history)
            future_te = te_emb[:, self.input_len:, :, :]
            hat_y = hat_y + future_te

        hat_y_out = self.end_emb(hat_y) + base_trend

        # 平均多层的注意力权重以获得更稳定的局部结构
        if len(attn_weights_list) > 0:
            attn_weights = torch.stack(attn_weights_list, dim=0).mean(dim=0)
        else:
            attn_weights = None

        return hat_y_out, attn_weights


def get_ablation_config(abl_type):
    """Return (ablation_flags, encoder_type) for the requested ablation."""
    base = {
        "wo_dual_frequency": False,
        "wo_sparse_spatial": False,
        "wo_bidirectional_cross": False,
        "wo_gated_fusion": False,
        "wo_temporal_graph": False,
        "wo_spatial_graph": False,
    }
    encoder_type = 'dual'
    if abl_type in ('full', 'dual'):
        return base, encoder_type
    mapping = {
        'wo_dual_freq': 'wo_dual_frequency',
        'wo_sparse': 'wo_sparse_spatial',
        'wo_bicross': 'wo_bidirectional_cross',
        'wo_gate': 'wo_gated_fusion',
        'wo_tem_graph': 'wo_temporal_graph',
        'wo_spa_graph': 'wo_spatial_graph',
    }
    key = mapping.get(abl_type)
    if key:
        cfg = base.copy()
        cfg[key] = True
        return cfg, encoder_type
    # Shared-encoder ablations defined in model.ablation
    shared = {'attn_shared', 'conv_shared'}
    if abl_type in shared:
        return base, abl_type
    raise ValueError(f"Unknown ablation type: {abl_type}")