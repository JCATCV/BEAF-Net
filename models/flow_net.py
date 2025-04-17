import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import reduce
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from .misc import constant_init

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 5)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super(BidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:

            feats[module_name] = []

            frame_idx = range(0, t)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]

            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feat_prop

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:  # second-order features
                        feat_n2 = feats[module_name][-2]
                        cond_n2 = feat_n2

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1) # condition information, cond(flow warped 1st/2nd feature)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1) # two order feat_prop -1 & -2
                    feat_prop = self.deform_align[module_name](feat_prop, cond)

                # fuse current features
                feat = [feat_current] + \
                    [feats[k][idx] for k in feats if k not in ['spatial', module_name]] \
                    + [feat_prop]

                feat = torch.cat(feat, dim=1)
                # embed current features
                feat_prop = feat_prop + self.backbone[module_name](feat)

                feats[module_name].append(feat_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class P3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_residual=0, bias=True):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size),
                                    stride=(1, stride, stride), padding=(0, padding, padding), bias=bias),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                    padding=(2, 0, 0), dilation=(2, 1, 1), bias=bias)
        )
        self.use_residual = use_residual

    def forward(self, feats):
        feat1 = self.conv1(feats)
        feat2 = self.conv2(feat1)
        if self.use_residual:
            output = feats + feat2
        else:
            output = feat2
        return output


class EdgeDetection(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, mid_ch=16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mid_layer_1 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mid_layer_2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1)
        )        

        self.l_relu = nn.LeakyReLU(0.01, inplace=True)

        self.out_layer = nn.Conv2d(mid_ch, out_ch, 1, 1, 0)

    def forward(self, flow):
        flow = self.projection(flow)
        edge = self.mid_layer_1(flow)
        edge = self.mid_layer_2(edge)
        edge = self.l_relu(flow + edge)
        edge = self.out_layer(edge)
        edge = torch.sigmoid(edge)
        return edge

# Implements global cross-frame attention for optical flow
# first, sampling a group of global optical flow. e.g., if the interval is set as 10, then 0, 10, 20, ... will be collected as global frames
# then, conduction cross-frame attention
class CrossAttention(nn.Module):
    def __init__(self, channel=128, head=4) -> None:
        super().__init__()
        self.proj_layer1 = nn.Linear(channel, channel)
        self.proj_layer2 = nn.Linear(channel, channel * 2)
        self.norm1 = nn.LayerNorm(channel)
        self.channel = channel
        self.head = head
        self.forward_layer = nn.Linear(channel, channel)

    def forward(self, flow_token, global_token):
        flow_token, global_token = self.norm1(flow_token), self.norm1(global_token)
        q, kv = self.proj_layer1(flow_token), self.proj_layer2(global_token.view(1,-1,self.channel))
        k, v = kv.chunk(2, -1) # q -> tx1620x128; k, v -> 1xt*162x128
        # q, k, v = q.reshape(-1, 1620, self.channel//4, 4).permute(0,3,1,2), k.reshape(1, -1, self.channel//4, 4).permute(0,3,1,2), \
            # v.reshape(1, -1, self.channel//4, 4).permute(0,3,1,2)
        # print("here: ", q.shape, k.shape, v.shape)
        attn_score = (q @ k.transpose(-1,-2))/(self.channel**0.5)
        attn_score = torch.softmax(attn_score, -1)
        v = self.forward_layer(attn_score @ v)
        return v



class GlobalCrossFrameAttention(nn.Module):
    def __init__(self, channel=128) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(channel)
        self.down_pool = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 2, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel * 2, channel, 1, 1, 0),
        )
        self.norm = nn.LayerNorm(channel)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channel * 4, channel)
        )

    def forward(self, flow, interval=10):
        t, c, h, w = flow.shape
        flow_token = flow.flatten(2).transpose(-1,-2) # t, h*w, c
        shortcut = flow_token
        global_flow_token = self.down_pool(flow[0:t:interval]).flatten(2).transpose(-1,-2) # t//10, h*w, c
        flow_token = self.cross_attention(flow_token, global_flow_token)
        flow_token = flow_token + shortcut
        shortcut = flow_token
        flow_token = self.mlp(self.norm(flow_token)) + shortcut
        return flow_token.transpose(-1,-2).reshape(t, c, h, w)


class GlobalLocalFusion(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
    def forward(self, global_flow, local_flow):
        flow = torch.cat((local_flow, global_flow), dim=1)
        return self.fuse(flow)

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, dropout=0.1):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b):
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, feat.size(2))
        feat = self.dropout(feat)
        return feat

class SoftComp(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.t2t = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size
        self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = self.t2t(feat) + self.bias[None]
        return feat

class AddPosEmb(nn.Module):
    def __init__(self, n, c):
        super(AddPosEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, n, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = n

    def forward(self, x):
        b, n, c = x.size()
        x = x.view(b, -1, self.num_vecs, c)
        x = x + self.pos_emb
        x = x.view(b, n, c)
        return x

class LocalRecurrentTransformer(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        # self.conv_pool = nn.Conv2d(channel, channel, 3, 2, 1, dilation=2)
        # self.model = LocalTransformer(channel)
        self.inpainter = LocalTransformer(channel)

    def forward(self, x):
        t, c, h, w = x.shape # in general 30x54 is set as spatial resolution
        # if t > 60:
        #     x1, x2 = x[:t//2], x[t//2:]
        #     x1, x2 = self.model(x1), self.model(x2)
        #     out = torch.cat((x1, x2), dim=0)
        # else:
        #     out = self.model(x)

        out = torch.zeros_like(x)
        for i in range(0, t, 10):
            if i > 5:
                input = torch.cat((x[i:i+10], out[i-5:i]), dim=0)
                output = self.inpainter(input)
                # print(input.shape, output.shape)
                out[i:i+10] = output[:-5]
            else:
                input = x[:10]
                out[:10] = self.inpainter(input)
            
        return out


class LocalTransformer(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        # kernel_size = (7, 7)
        # padding = (3, 3)
        # stride = (3, 3)

        kernel_size = (5, 5)
        padding = (2, 2)
        stride = (2, 2)
        
        output_size = (30, 54)
        
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)
        self.patch2token = SoftSplit(channel, channel * 2, kernel_size, stride, padding)
        self.token2patch = SoftComp(channel, channel * 2, output_size, kernel_size, stride, padding)
        self.add_pos_emb = AddPosEmb(n_vecs, channel * 2)
        self.transformer1 = TransformerBlock(channel * 2)
        self.transformer2 = TransformerBlock(channel * 2)
        # print(self.transformer1.weight.grad)

    def forward(self, x):
        t, c, h, w = x.shape 
        trans_feat = self.patch2token(x.view(-1, c, h, w), 1)
        trans_feat = self.add_pos_emb(trans_feat)
        # print("here: ", trans_feat.shape)
        trans_feat = self.transformer1(trans_feat)
        trans_feat = self.transformer2(trans_feat)
        trans_feat = self.token2patch(trans_feat, t)
        x = x + trans_feat
        return x

import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill_(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        b, n, c = x.size()
        c_h = c // self.head
        key = self.key_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = self.query_embedding(x)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = self.value_embedding(x)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        # print(self.norm1.weight.grad)

    def forward(self, input):
        x = self.norm1(input)
        x = input + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return x

class LocalAggregator(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel * 3, channel * 2, 3, 1, 2, dilation=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel * 2, channel * 2, 3, 1, 2, dilation=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel * 2, channel, 3, 1, 2, dilation=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
    def forward(self, x):

        return self.fusion_conv(x)

class FlowCompletionNet(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.downsample = nn.Sequential(
                        nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2), 
                                        padding=(0, 2, 2), padding_mode='replicate'),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder1 = nn.Sequential(
            P3DBlock(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            P3DBlock(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 4x

        self.encoder2 = nn.Sequential(
            P3DBlock(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            P3DBlock(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 8x

        self.mid_dilation = nn.Sequential(
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)), # p = d*(k-1)/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # feature propagation module
        self.feat_prop_module = BidirectionalPropagation(128)

        # self.feat_prop_module2 = BidirectionalPropagation(128)
        # self.transformer = LocalRecurrentTransformer(128)
        # self.transformer = nn.Conv2d(128,128,3,1,1)

        # self.local_aggregation = LocalAggregator(128)


        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(128, 64, 3, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 4x

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 32, 3, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 2x

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(32, 2, 3, 1)
        )

        # edge loss
        self.new_edgeDetector = EdgeDetection(in_ch=2, out_ch=1, mid_ch=64)


        # self.conv_fusion = nn.Sequential(
        #     P3DBlock(128, 128, 3, 1, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     P3DBlock(128, 128, 3, 1, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     P3DBlock(128, 128, 3, 1, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # self.global_attention = GlobalCrossFrameAttention(128)

        # self.g2l_fusion = GlobalLocalFusion(128)

        # Need to initial the weights of MSDeformAttn specifically
        for m in self.modules():
            if isinstance(m, SecondOrderDeformableAlignment):
                m.init_offset()

        if model_path is not None:
            print('Pretrained flow completion model has loaded...')
            ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=False)


    def forward(self, masked_flows, masks, interval):
        # masked_flows: b t-1 2 h w
        # masks: b t-1 2 h w
        b, t, _, h, w = masked_flows.size()
        masked_flows = masked_flows.permute(0,2,1,3,4)
        masks = masks.permute(0,2,1,3,4)

        inputs = torch.cat((masked_flows, masks), dim=1)
        
        x = self.downsample(inputs)

        feat_e1 = self.encoder1(x)
        feat_e2 = self.encoder2(feat_e1) # b c t h w
        feat_mid = self.mid_dilation(feat_e2) # b c t h w
        feat_mid = feat_mid.permute(0,2,1,3,4) # b t c h w


        feat_prop = self.feat_prop_module(feat_mid)

        # feat_prop = feat_mid.view(-1, 128, h//8, w//8) # b*t c h w
        
        # feat_for_shift = torch.cat((feat_prop[1:], feat_prop[-1].unsqueeze(0)), dim=0)
        # feat_back_shift = torch.cat((feat_prop[:-1], feat_prop[0].unsqueeze(0)), dim=0)
        # feat_cat = torch.cat((feat_back_shift, feat_prop, feat_for_shift), dim=1)
        # # print("here: ", feat_cat.shape)
        # feat_prop = feat_prop + self.local_aggregation(feat_cat)

        # feat_prop = self.transformer(feat_prop)

        # feat_prop = feat_mid.view(-1, 128, h//8, w//8)
        feat_prop = feat_prop.view(-1, 128, h//8, w//8)
        _, c, _, h_f, w_f = feat_e1.shape
        feat_e1 = feat_e1.permute(0,2,1,3,4).contiguous().view(-1, c, h_f, w_f) # b*t c h w
        feat_d2 = self.decoder2(feat_prop) + feat_e1

        _, c, _, h_f, w_f = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(-1, c, h_f, w_f) # b*t c h w

        feat_d1 = self.decoder1(feat_d2)

        flow = self.upsample(feat_d1)
        
        edge = self.new_edgeDetector(flow)
        edge = edge.view(b, t, 1, h, w)


        flow = flow.view(b, t, 2, h, w)

        return flow, edge
        

    def forward_bidirect_flow(self, masked_flows_bi, masks, interval=10):
        """
        Args:
            masked_flows_bi: [masked_flows_f, masked_flows_b] | (b t-1 2 h w), (b t-1 2 h w)
            masks: b t 1 h w
        """
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bi[0] * (1-masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1-masks_backward)
        
        # -- completion --
        # forward
        pred_flows_forward, pred_edges_forward = self.forward(masked_flows_forward, masks_forward, interval)

        # backward
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        pred_flows_backward, pred_edges_backward = self.forward(masked_flows_backward, masks_backward, interval)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        
        pred_edges_backward = torch.flip(pred_edges_backward, dims=[1])

        return [pred_flows_forward, pred_flows_backward], [pred_edges_forward, pred_edges_backward]


    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1-masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1-masks_backward)

        return pred_flows_forward, pred_flows_backward
