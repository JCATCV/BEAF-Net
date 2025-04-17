import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat



class SpatailDetailEnhancement(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super(self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        
        self.conv1 = nn.Conv2d(2, 1, 3, 2, 1)
        self.conv2 = nn.Sequential(nn.GELU(), nn.Conv2d(1, 1, 3, 2, 1))

        self.conv3 = nn.Sequential(nn.Conv2d(42, 40, 3, 1, 1))

        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size']) # 49

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])
        
        x0 = torch.cat((x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]), 1)
        y1 = self.conv1(x0)
        y2 = self.conv2(y1)
        y1 = F.interpolate(y1, scale_factor=2, align_corners=True, mode="bilinear")
        y2 = F.interpolate(y2, scale_factor=4, align_corners=True, mode="bilinear")
        
        x = self.conv3(torch.cat((x, y1, y2), dim=1))
        

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.fc2(x)
        return x


def window_partition(x, window_size, n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C//n_head)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows

class Modulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.squeeze = nn.Conv2d(512, 128, 1, 1, 0)
        self.forward_layer = nn.Conv2d(128, 128, 3, 1, 1)
        self.unsqueeze = nn.Conv2d(128, 512, 1, 1, 0)
        self.non_linear = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, m):
        b, t, h, w, c = x.shape
        b, l_t, h, w, _ = m.shape
        l_x = x[:, :l_t]
        l_x = (l_x + m).squeeze(0).permute(0,3,1,2)
        l_x = self.squeeze(l_x)
        l_x = self.non_linear(l_x)
        l_x = self.forward_layer(l_x)
        confidence_score = torch.sigmoid(l_x)
        l_x = l_x * confidence_score
        l_x = self.unsqueeze(l_x)
        x[:, :l_t] = l_x.permute(0,2,3,1).unsqueeze(0)
        return x
    
class ErrorAwareSelfAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0., 
                pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        self.test = False
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
            self.pool_layer_mask = nn.Conv2d(1, 1, kernel_size=ks, stride=stride, padding=(0, 0))
            self.pool_layer_mask.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer_mask.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))
        self.key_modulator = Modulator()
        self.value_modulator = Modulator()

    def forward(self, x, mask=None, T_ind=None):
        b, t, h, w, c = x.shape # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0] # 20
        new_w = n_ww * self.window_size[1] # 36
        pad_r = new_w - w
        pad_b = new_h - h
        
        # reverse order

        if pad_r > 0 or pad_b > 0:
            x = F.pad(x,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 
            mask = F.pad(mask,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        m = mask

        k = self.key_modulator(k, mask, name="k")
        v = self.value_modulator(v, mask, name="v")

        win_q = window_partition(q.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_mask = window_partition(mask.contiguous(), self.window_size, 1).view(b, n_wh*n_ww, 1, t, w_h*w_w, 1)
        raw_mask = win_mask
        if any(i > 0 for i in self.expand_size):
            (k_tl, v_tl) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (k, v))
            (k_br, v_br) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (k, v))
            mask_tl = torch.roll(mask, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3))
            mask_tr = torch.roll(mask, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3))
            mask_bl = torch.roll(mask, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3))
            mask_br = torch.roll(mask, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head), 
                (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head), 
                (v_tl, v_tr, v_bl, v_br))
            (mask_tl_windows, mask_tr_windows, mask_bl_windows, mask_br_windows) = map(
                lambda a: window_partition(a, self.window_size, 1).view(b, n_wh*n_ww, 1, t, w_h*w_w, 1), 
                (mask_tl, mask_tr, mask_bl, mask_br))
            
            rool_k = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 4).contiguous()
            rool_v = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 4).contiguous() # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            rool_mask = torch.cat((mask_tl_windows, mask_tr_windows, mask_bl_windows, mask_br_windows), 4).contiguous()
            
            # mask out tokens in current window
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            rool_mask = rool_mask[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(b, n_wh*n_ww, self.n_head, t, roll_N, c // self.n_head)
            rool_v = rool_v.view(b, n_wh*n_ww, self.n_head, t, roll_N, c // self.n_head)
            rool_mask = rool_mask.view(b, n_wh*n_ww, 1, t, roll_N, 1)
            win_k = torch.cat((win_k, rool_k), dim=4)
            win_v = torch.cat((win_v, rool_v), dim=4)
            win_mask = torch.cat((win_mask, rool_mask), dim=4)
            
        else:
            win_k = win_k
            win_v = win_v
            win_mask = win_mask
        
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b*t, new_h, new_w, c).permute(0,3,1,2))
    
            pool_mask = self.pool_layer_mask(mask.view(b*t, new_h, new_w, 1).permute(0,3,1,2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0,2,3,1).view(b, t, p_h, p_w, c)
            pool_mask = pool_mask.permute(0,2,3,1).view(b, t, p_h, p_w, 1)
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_wh*n_ww, 1, 1, 1, 1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_k = pool_k.view(b, n_wh*n_ww, t, p_h, p_w, self.n_head, c_head).permute(0,1,5,2,3,4,6)
            pool_k = pool_k.contiguous().view(b, n_wh*n_ww, self.n_head, t, p_h*p_w, c_head)
            win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_wh*n_ww, 1, 1, 1, 1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_v = pool_v.view(b, n_wh*n_ww, t, p_h, p_w, self.n_head, c_head).permute(0,1,5,2,3,4,6)
            pool_v = pool_v.contiguous().view(b, n_wh*n_ww, self.n_head, t, p_h*p_w, c_head)
            win_v = torch.cat((win_v, pool_v), dim=4)
            # pool_m
            pool_mask = pool_mask.unsqueeze(1).repeat(1, n_wh*n_ww, 1, 1, 1, 1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_mask = pool_mask.view(b, n_wh*n_ww, t, p_h, p_w, 1, 1).permute(0,1,5,2,3,4,6)
            pool_mask = pool_mask.contiguous().view(b, n_wh*n_ww, 1, t, p_h*p_w, 1)
            win_mask = torch.cat((win_mask, pool_mask), dim=4)

        out = torch.zeros_like(win_q)
        out_mask =torch.zeros_like(raw_mask)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh*n_ww)
        mask = torch.sum(mask, dim=1) # [b, n_wh*n_ww]

        for i in range(win_q.shape[0]):
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.n_head, t*w_h*w_w, c_head)
                win_k_t = win_k[i, mask_ind_i] 
                win_v_t = win_v[i, mask_ind_i] 
                win_mask_t = win_mask[i, mask_ind_i]

                if T_ind is not None:
                    # key
                    win_k_t = win_k_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                    # value
                    win_v_t = win_v_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                    # mask
                    win_mask_t = win_mask_t[:, :, T_ind.view(-1)].view(mask_n, 1, -1, 1)
                else:
                    win_k_t = win_k_t.view(n_wh*n_ww, self.n_head, t*w_h*w_w, c_head)
                    win_v_t = win_v_t.view(n_wh*n_ww, self.n_head, t*w_h*w_w, c_head)
                    win_mask_t = win_mask_t.view(n_wh*n_ww, 1, t*w_h*w_w, 1)

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = att_t + (win_mask_t * -100.).transpose(-1,-2).expand_as(att_t)
                att_t = F.softmax(att_t, dim=-1) 
                y_t = att_t @ win_v_t 
                
                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h*w_w, c_head)
                y_t_mask = att_t.mean(1, True) @ win_mask_t
                out_mask[i, mask_ind_i] = y_t_mask.view(-1, 1, t, w_h*w_w, 1)
                
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, :w_h*w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, :w_h*w_w]
            win_mask_s = win_mask[i, unmask_ind_i, :, :, :w_h*w_w]
            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))

            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s
            out_mask[i, unmask_ind_i] = att_s.mean(1, True) @ win_mask_s
        
        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)

        out_mask = out_mask.view(b, n_wh, n_ww, 1, t, w_h, w_w, 1)
        out_mask = out_mask.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, 1)


        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]
            out_mask = out_mask[:, :, :h, :w, :]

        # output projection
        out = self.proj_drop(self.proj(out))
        

        return out, out_mask


class ErrorAwareTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size,
                norm_layer=nn.LayerNorm, t2t_params=None):
        super().__init__()

        self.window_size = window_size
        self.attention = ErrorAwareSelfAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = SpatailDetailEnhancement(dim, t2t_params=t2t_params)

        self.back_attention = ErrorAwareSelfAttention(dim, n_head, window_size, pool_size)
        self.norm = norm_layer(dim)
        self.norm0 = norm_layer(dim)
        self.mlp0 = SpatailDetailEnhancement(dim, t2t_params=t2t_params)

        self.bidirectional_proj = nn.Linear(512, 512)
        self.bidirectional_fusion = nn.Conv2d(4, 1, 3, 1, 1, bias=False)

    def forward(self, x, m, l_t, fold_x_size, mask=None, T_ind=None, iteration=None, raw_img=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape 
        m_t = m.size(1)
        T_ind_x, T_ind_m = T_ind

        # backward search
        bx = torch.cat((m, x[:, :l_t]), 1)
        bm = torch.cat((torch.zeros_like(m).mean(-1, keepdim=True), mask[:, :l_t]), 1)
        shortcut = bx
        bx = self.norm(bx)
        
        att_bx, att_bm = self.back_attention(bx, bm, T_ind_m, iteration = iteration, raw_img = raw_img)
        bx = att_bx + shortcut
        by = self.norm0(bx)
        bx = bx + self.mlp0(by.view(B, (l_t + m_t) * H * W, C), fold_x_size).view(B, l_t + m_t, H, W, C)


        shortcut = x
        x = self.norm1(x)
        att_x, att_m = self.attention(x, mask, T_ind_x, iteration = iteration, raw_img = raw_img)

        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        x1, x2 = x[:, :l_t], bx[:, m_t:]
        fusion_map = torch.cat((x1.mean(-1, keepdim=True), x1.max(-1, keepdim=True)[0], x2.mean(-1, keepdim=True), x2.max(-1, keepdim=True)[0]), -1)
        fusion_map = self.bidirectional_fusion(fusion_map.squeeze(0).permute(0,3,1,2))
        fusion_map = torch.sigmoid(fusion_map.squeeze(1).unsqueeze(0).unsqueeze(-1))
        x[:, :l_t] = self.bidirectional_proj(x[:, :l_t].clone()* fusion_map  + bx[:, m_t:] * (1. - fusion_map))
        att_m[:, :l_t] = att_m[:, :l_t].clone() * fusion_map + att_bm[:, m_t:] * (1. - fusion_map)
        return x, bx[:, :m_t], att_m
    


class ErrorAwareTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None):
        super().__init__()
        blocks = []
        for i in range(depths):
             blocks.append(
                ErrorAwareTransformer(dim, n_head, window_size, pool_size, t2t_params=t2t_params)
             )
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths
    def forward(self, x, m, l_t, fold_x_size, mask=None, t_dilation=2, raw_img=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        T = x.size(1)
        T_ind_x = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)
        T_ind_m = [torch.arange(i, l_t + m.size(1), t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)
        upd_mask_iter = []
        for i in range(0, self.depths):
            x, m, mask = self.transformer[i](x, m, l_t, fold_x_size, mask, (T_ind_x[i], T_ind_m[i]), iteration=i, raw_img=raw_img)
            upd_mask_iter.append(mask.squeeze(0).squeeze(-1).detach().cpu().numpy()[0])
        
        return x, m, mask
