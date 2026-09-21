import numpy as np
import torch
import torch.nn as nn

from model.models import TemAttn, TemConv, SparseSTFExtraction


class FSTEncoderSharedAttn(nn.Module):
    """Ablation variant 1: both low/high-frequency branches use Temporal Attention.

    Unlike FSTEncoder (which uses Attention for low-freq and Convolution for
    high-freq), this variant applies the same Temporal Attention module to both
    branches with independent parameters (no weight sharing).
    """

    def __init__(self, heads, dims, samples, levels, localadj, spawave, temwave):
        super(FSTEncoderSharedAttn, self).__init__()
        # Two independent Temporal Attention modules (not shared)
        self.temporal_att_l = TemAttn(heads, dims)
        self.temporal_att_h = TemAttn(heads, dims)
        # Spatial modules remain independent per branch
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
        return: (xl, xh)
        """
        xl = self.temporal_att_l(xl, te)
        xh = self.temporal_att_h(xh, te)

        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec,
                                         self.tem_eigvalue, self.tem_eigvec)
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec,
                                         self.tem_eigvalue, self.tem_eigvec)
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        return xl, xh


class FSTEncoderSharedConv(nn.Module):
    """Ablation variant 2: both low/high-frequency branches use Temporal Causal Conv.

    Unlike FSTEncoder (which uses Attention for low-freq and Convolution for
    high-freq), this variant applies the same TemConv module to both branches
    with independent parameters (no weight sharing).
    """

    def __init__(self, heads, dims, samples, levels, localadj, spawave, temwave):
        super(FSTEncoderSharedConv, self).__init__()
        # Two independent TemConv modules (not shared)
        self.temporal_conv_l = TemConv(heads * dims, levels=levels)
        self.temporal_conv_h = TemConv(heads * dims, levels=levels)
        # Spatial modules remain independent per branch
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
        return: (xl, xh)
        """
        xl = self.temporal_conv_l(xl)
        xh = self.temporal_conv_h(xh)

        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec,
                                         self.tem_eigvalue, self.tem_eigvec)
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec,
                                         self.tem_eigvalue, self.tem_eigvec)
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        return xl, xh


# Registry mapping ablation name -> encoder class
ABLATION_ENCODERS = {
    'attn_shared': FSTEncoderSharedAttn,
    'conv_shared': FSTEncoderSharedConv,
}