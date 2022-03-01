"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
-----------------------------------
Modified by Zhuofan Xia for Vision Transformer with Deformable Attention
"""

import math

import jittor as jt
import jittor.nn as nn
from jimm.models.registry import register_model
from jimm.models.layers import DropPath, to_2tuple, trunc_normal_


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = jt.zeros([(2 * Wh - 1) * (2 * Ww - 1), heads])
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = jt.arange(self.window_size[0])
        coords_w = jt.arange(self.window_size[1])
        coords = jt.stack(jt.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = jt.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
    def execute(self, x, mask=None):

        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]
        N = self.window_size[0] * self.window_size[1]
        M = r1 * r2
        h = self.heads
        hc = self.head_dim

        x = x.reshape(B, C, r1, self.window_size[0], r2, self.window_size[1])
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, M, N, C)
        x_total = x

        qkv = self.proj_qkv(x_total)
        q, k, v = jt.chunk(qkv, 3, dim=3)
        q = q * self.scale

        B_ = B * M
        q, k, v = [t.reshape(B_, N, h, hc).permute(0, 2, 1, 3).reshape(B_ * h, N, hc) for t in [q, k, v]] # B_h, N, hc
        
        attn = nn.bmm_transpose(q, k) # B_h, N, N

        attn = attn.reshape(B_, h, N, N)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # N, N, h
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # h, N, N
        attn = attn + relative_position_bias.unsqueeze(0) # B_, h, N, N

        if mask is not None:
            nW, ww, _ = mask.size() # nW, N, N
            assert nW == M and ww == N
            attn = attn.reshape(B, M, h, N, N) + mask.reshape(1, M, 1, N, N)
            attn = attn.reshape(B_, h, N, N)
        
        attn = nn.softmax(attn, dim=3)
        
        attn = self.attn_drop(attn)

        x = nn.bmm(attn.reshape(B_ * h, N, N), v) # B_h, N, hc
        
        x = x.reshape(B_, h, N, hc).permute(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj_drop(self.proj_out(x)) # B_, N, C

        x = x.reshape(B, r1, r2, self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)
        
        return x, None, None

class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = jt.zeros(self.fmap_size)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1

        H, W = self.fmap_size
        r1, r2 = H // self.window_size[0], W // self.window_size[1]
        M = r1 * r2 
        N = self.window_size[0] * self.window_size[1]
        mask_windows = img_mask.reshape(r1, self.window_size[0], r2, self.window_size[1]).permute(0, 2, 1, 3).reshape(M, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # M N N
        self.attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
      
    def execute(self, x):

        shifted_x = jt.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().execute(shifted_x, self.attn_mask)
        x = jt.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None

class DAttentionBaseline(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        
        if self.q_h == 14 or self.q_w == 14 or self.q_h == 24 or self.q_w == 24:
            kk = 5
        elif self.q_h == 7 or self.q_w == 7 or self.q_h == 12 or self.q_w == 12:
            kk = 3
        elif self.q_h == 28 or self.q_w == 28 or self.q_h == 48 or self.q_w == 48:
            kk = 7
        elif self.q_h == 56 or self.q_w == 56 or self.q_h == 96 or self.q_w == 96:
            kk = 9

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = jt.zeros([self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w])
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = jt.zeros([self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1])
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
    
    def _get_ref_points(self, H_key, W_key, B):
        
        ref_y, ref_x = jt.meshgrid(
            jt.linspace(0.5, H_key - 0.5, H_key), 
            jt.linspace(0.5, W_key - 0.5, W_key)
        )
        ref = jt.stack((ref_y, ref_x), -1)
        ref[..., 1] = ref[..., 1].divide(W_key).multiply(2).subtract(1)
        ref[..., 1] = ref[..., 0].divide(H_key).multiply(2).subtract(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g, H, W, 2
        
        return ref

    def execute(self, x):

        B, C, H, W = x.size()
        g = self.n_groups
        gc = self.n_group_channels
        q = self.proj_q(x)
        h = self.n_heads
        hc = self.n_head_channels
        gh = self.n_group_heads
        q_off = q.reshape(B * g, gc, H, W)
        
        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = jt.array([1.0 / Hk, 1.0 / Wk]).reshape(1, 2, 1, 1)
            offset = offset.tanh().multiply(offset_range).multiply(self.offset_range_factor)
        
        offset = offset.permute(0, 2, 3, 1) # B * g Hg Wg 2
        
        reference = self._get_ref_points(Hk, Wk, B)
            
        if self.no_off:
            offset.fill_(0.0)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        x_sampled = nn.grid_sample(
            input=x.reshape(B * g, gc, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * h, hc, H * W).permute(0, 2, 1)
        k = self.proj_k(x_sampled).reshape(B * h, hc, n_sample).permute(0, 2, 1)
        v = self.proj_v(x_sampled).reshape(B * h, hc, n_sample).permute(0, 2, 1)
        # q, k, v: Bh HW(Ns) hc
        
        attn = nn.bmm_transpose(q, k) # Bh, HW, Ns
        attn = attn.multiply(self.scale)
        
        if self.use_pe:
            
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * h, hc, H * W).permute(0, 2, 1)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * h, H * W, n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B)
                
                displacement = (q_grid.reshape(B * g, H * W, 2).unsqueeze(2) - pos.reshape(B * g, n_sample, 2).unsqueeze(1)).multiply(0.5)
                
                attn_bias = nn.grid_sample(
                    input=rpe_bias.reshape(B * g, gh, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, gh, HW, Ns
                
                attn_bias = attn_bias.reshape(B * h, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = nn.softmax(attn, dim=2)
        attn = self.attn_drop(attn) # Bh, HW, Ns
        
        out = nn.bmm(attn, v) # Bh HW hc
        
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        y = self.proj_drop(self.proj_out(out))
        
        return y, pos.reshape(B, g, Hk, Wk, 2), reference.reshape(B, g, Hk, Wk, 2)

class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop))
    
    def execute(self, x):
       
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.chunk(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x

class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(drop)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def execute(self, x):
        
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        
        return x

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def execute(self, x):
        
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        return x.permute(0, 2, 1).reshape(B, C, H, W)



class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, stride, offset_range_factor, 
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def execute(self, x):
        
        x = self.proj(x)
        
        positions = []
        references = []
        for d in range(self.depths):

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references

class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], 
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1], offset_range_factor=[1, 2, 3, 4], 
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 **kwargs):
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        ) 

        img_size = img_size // patch_size
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], 
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i])
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
           
        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reset_parameters()
    
    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    def execute(self, x):
        
        x = self.patch_proj(x)
        positions = []
        references = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
        x = self.cls_norm(x)
        x = self.avg_pool(x)
        x = jt.flatten(x, 1)
        x = self.cls_head(x)
        
        return x, positions, references


@register_model
def dat_tiny():

    return DAT(
        dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D','L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes= [False, False, False, False],
        strides= [-1, -1, 1, 1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_path_rate=0.2
    )


@register_model
def dat_small():

    return DAT(
        dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 18, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D','L', 'D', 'L', 'D', 'L', 'D','L', 'D', 'L', 'D', 'L', 'D','L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes= [False, False, False, False],
        strides= [-1, -1, 1, 1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_path_rate=0.3
    )

@register_model
def dat_base():

    return DAT(
        dim_stem=128, dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D','L', 'D', 'L', 'D', 'L', 'D','L', 'D', 'L', 'D', 'L', 'D','L', 'D'], ['L', 'D']],
        heads=[4, 8, 16, 32],
        groups=[-1, -1, 4, 8],
        use_pes=[False, False, True, True],
        dwc_pes= [False, False, False, False],
        strides= [-1, -1, 1, 1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_path_rate=0.5
    )