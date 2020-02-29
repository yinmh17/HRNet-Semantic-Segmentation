import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math
from .modules.mat_expand import MatExpand


class _MultiHeadNonLocalNd(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, num_head,
                 distance, distance_delta, distance_mean,
                 pos_embed_dim, pos_feat_dim, pos_beta,
                 use_saliency, saliency_alpha, use_gn, lr_mult, whiten_type, temp, nowd):
        ### example, use_saliency = "A", saliency_alpha = 1.0 or 0.5, pos_feat_dim = 256
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
                avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
                avg_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
                avg_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
                avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
                avg_pool = None
            bn_nd = nn.BatchNorm1d

        super(_MultiHeadNonLocalNd, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.num_head = num_head
        self.distance = distance
        self.distance_delta = distance_delta
        self.distance_mean = distance_mean
        assert distance in ['dot', 'cosine', 'l2', 'renorm_cosine']
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else None
        self.scale = math.sqrt(planes / num_head)

        # "" means no saliency map
        # "B" means saliency before softmax
        # "A" means saliency after softmax
        assert len(use_saliency) == 1 or len(use_saliency) == 0, '{} must be 0 or 1'.format(len(use_saliency))
        self.use_saliency = use_saliency

        if self.use_saliency in ["Q", "A"]:
            self.conv_query_saliency = conv_nd(inplanes, num_head, kernel_size=1)
        else:
            self.conv_query_saliency = None
        if self.use_saliency in ["K", "A", "S", "V"]:
            self.conv_key_saliency = conv_nd(inplanes, num_head, kernel_size=1)
        else:
            self.conv_key_saliency = None

        self.saliency_alpha = saliency_alpha

        self.pos_embed_dim = pos_embed_dim
        self.pos_feat_dim = pos_feat_dim
        self.pos_beta = pos_beta

        self.whiten_type = whiten_type
        self.temp = temp
        self.nowd = nowd
        self.use_gn = use_gn
        self.weight_init_scale = 1.0
        self.with_nl = True

        if self.pos_feat_dim > 0:
            #self.conv_pos = nn.Conv2d(pos_embed_dim, pos_feat_dim, kernel_size=1)
            #self.conv_pos_out = nn.Conv2d(pos_feat_dim, num_head, kernel_size=1)
            self.conv_pos_out = nn.Conv2d(pos_embed_dim, num_head, kernel_size=1)
            self.mapping = MatExpand()

        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = avg_pool

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)
        if len(self.nowd)>0:
            self.reset_weight_and_weight_decay()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
            if self.use_saliency:
                if m == self.conv_query_saliency or m == self.conv_key_saliency:
                    init.normal_(m.weight, 0, 0.01)
                    init.constant_(m.bias, -2.3)
                    m.inited = True                 
            if self.pos_feat_dim > 0:
                if m == self.conv_pos_out:
                    init.constant_(m.weight, 0)
                    init.constant_(m.bias, 1.0)
                    m.inited = True
        if self.use_gn:
            init.constant_(self.norm.weight, 0)
            init.constant_(self.norm.bias, 0)
            self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
                
    def reset_weight_and_weight_decay(self):
        if self.with_nl:
            init.normal_(self.conv_query.weight, 0, 0.01 * self.weight_init_scale)
            init.normal_(self.conv_key.weight, 0, 0.01 * self.weight_init_scale)
            if 'nl' in self.nowd:
                self.conv_query.weight.wd = 0.0
                self.conv_query.bias.wd = 0.0
                self.conv_key.weight.wd = 0.0
                self.conv_key.bias.wd = 0.0
        if self.use_saliency and 'gc' in self.nowd:
            self.conv_key_saliency.weight.wd = 0.0
            self.conv_key_saliency.bias.wd = 0.0
        if 'value' in self.nowd:
            self.conv_value.weight.wd = 0.0
            # self.conv_value.bias.wd=0.0

    def extract_position_embedding(self, embed_dim, feat_dim, H, W, wave_length=1000):
        # [2H-1, 2W-1, 1, 1]
        y_range = torch.arange(-H + 1, H, dtype=torch.float).view(-1, 1, 1, 1).repeat(1, 2 * W - 1, 1, 1)
        x_range = torch.arange(-W + 1, W, dtype=torch.float).view(1, -1, 1, 1).repeat(2 * H - 1, 1, 1, 1)

        # [2H-1, 2W-1, 2, 1]
        pos_mat = torch.cat((y_range, x_range), 2)
        # add tanh with beta
        #pos_mat = torch.tanh(pos_mat / self.pos_beta) * 100

        # [1, 1, 1, embed_dim/4]
        feat_range = torch.arange(0, embed_dim / 4, dtype=torch.float)
        feat_range = feat_range * (4.0 / feat_dim)
        dim_mat = torch.pow(wave_length, feat_range).view(1, 1, 1, -1)

        # [2H-1, 2W-1, 2, embed_dim/4]
        div_mat = pos_mat / dim_mat

        # [2H-1, 2W-1, embed_dim]
        feat = torch.cat((div_mat.sin(), div_mat.cos()), dim=3).view(2 * H - 1, 2 * W - 1, feat_dim)

        # [1, embed_dim, 2H-1, 2W-1]
        feat = feat.unsqueeze(0).transpose(0, 3).view(1, feat_dim, 2 * H - 1, 2 * W - 1).cuda()

        # embedding.block_gradient()

        ### extract position feature
        # [1, feat_dim, 2H-1, 2W-1]
        #feat = self.conv_pos(feat)
        #feat = self.relu(feat)

        # [1, n_Head, 2H-1, 2W-1]
        feat = self.conv_pos_out(feat)
        feat = self.relu(feat)

        ### Use kernel to extract specific position map
        # [1, n_Head, H * W, H * W]
        feat = self.mapping(feat)

        # [1, n_Head, H * W, H' * W']
        feat = feat.view(self.num_head, H * W, H, W)
        if self.avg_pool is not None:
            feat = self.avg_pool(feat).view(1, self.num_head, H * W, int(H / 2) * int(W / 2))
        else:
            feat = feat.view(1, self.num_head, H * W, H * W)

        return feat

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        if "Q" in self.use_saliency or "A" in self.use_saliency:
            # [N, nHead, T, H, W]
            saliency = torch.sigmoid(self.conv_query_saliency(x)).pow(self.saliency_alpha)
            # [N x nHead, T x H x W, 1]
            saliency = saliency.view(saliency.size(0) * saliency.size(1), -1, 1)

        if "S" in self.use_saliency:
            saliency_down = self.conv_key_saliency(input_x)
            # [N, nHead, T * H' * W']
            saliency_down = saliency_down.view(saliency_down.size(0), saliency_down.size(1), -1)
            saliency_down = F.softmax(saliency_down, dim=2)
            # [N * nHead, T * H' * W']
            saliency_down = saliency_down.view(saliency_down.size(0) * saliency_down.size(1), -1)
            # [N * nHead, 1, T * H' * W']
            saliency_down = saliency_down.unsqueeze(1)

        if "K" in self.use_saliency or "A" in self.use_saliency:
            # [N, nHead, T, H', W']
            saliency_down = torch.sigmoid(self.conv_key_saliency(input_x)).pow(self.saliency_alpha)
            # [N x nHead, 1, T x H' x W']
            saliency_down = saliency_down.view(saliency_down.size(0) * saliency_down.size(1), 1, -1)

        if "V" in self.use_saliency:
            # [N, nHead, T, H', W']
            saliency_down = self.conv_key_saliency(input_x)
            # [N x nHead, 1, T x H' x W']
            saliency_down = saliency_down.view(saliency_down.size(0) * saliency_down.size(1), 1, -1)

        # [N x nHead, C'/nHead, T x H x W]
        query = query.view(query.size(0), self.num_head, int(query.size(1) / self.num_head), -1)
        query = query.view(query.size(0) * self.num_head, *query.size()[2:])

        # [N x nHead, C'/nHead, T x H' x W']
        key = key.view(key.size(0), self.num_head, int(key.size(1) / self.num_head), -1)
        key = key.view(key.size(0) * self.num_head, *key.size()[2:])

        # [N x nHead, C/nHead, T x H' x W']
        value = value.view(value.size(0), self.num_head, int(value.size(1) / self.num_head), -1)
        value = value.view(value.size(0) * self.num_head, *value.size()[2:])

        if 'in' in self.whiten_type:
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean

        if self.distance_mean:
            query = query - query.mean(dim=1, keepdim=True)
            key = key - key.mean(dim=1, keepdim=True)
        if self.distance == 'l2':
            sim_map = -(query.pow(2).sum(1).unsqueeze(2)
                        - 2 * torch.bmm(query.transpose(1, 2), key)
                        + key.pow(2).sum(1).unsqueeze(1)).pow(0.5)
            sim_map = sim_map * self.distance_delta
        elif self.distance == 'cosine':
            sim_map = torch.bmm(query.transpose(1, 2), key) / (1e-5 + query.norm(dim=1).unsqueeze(2)
                                                               * key.norm(dim=1).unsqueeze(1))
            sim_map = sim_map * self.distance_delta
        elif self.distance == 'renorm_cosine':
            sim_map = torch.bmm(query.transpose(1, 2), key) / (1e-5 + query.norm(dim=1).unsqueeze(2)
                                                               * key.norm(dim=1).unsqueeze(1))
            sim_map = sim_map * self.distance_delta
            sim_map = F.softmax(sim_map, dim=1)
            sim_map = sim_map / (1e-5 + sim_map.sum(dim=2, keepdim=True))
        else:
            # [N x nHead, T x H x W, T x H' x W']
            sim_map = torch.bmm(query.transpose(1, 2), key)
            # [N x nHead, T x H x W, T x H' x W']
            if len(self.nowd)==0:
                sim_map = sim_map / self.scale
                sim_map = sim_map / self.temp

        if self.pos_feat_dim > 0:
            pos_map = self.extract_position_embedding(self.pos_embed_dim, self.pos_feat_dim, *x.size()[2:])
            # [N, nHead, T x H x W, T x H' x W']
            sim_map = sim_map.view(-1, self.num_head, *sim_map.size()[1:]) + pos_map
            # [N x nHead, T x H x W, T x H' x W']
            sim_map = sim_map.view(sim_map.size(0) * self.num_head, *sim_map.size()[2:])

        if "K" in self.use_saliency:
            sim_map = sim_map + torch.log(1e-5 + saliency_down)

        if "Q" in self.use_saliency:
            sim_map = sim_map + torch.log(1e-5 + saliency)

        if "V" in self.use_saliency:
            sim_map = sim_map + saliency_down

        if 'renorm' in self.distance:
            att_map = sim_map
        else:
            att_map = self.softmax(sim_map)

        if "A" in self.use_saliency:
            att_map = att_map * saliency * saliency_down

        if "S" in self.use_saliency:
            att_map = att_map + saliency_down

        # [N x nHead, T x H x W, C/nHead]
        out = torch.bmm(att_map, value.transpose(1, 2))
        # [N x nHead, C/nHead, T x H x W]
        out = out.transpose(1, 2)
        # [N, C/nHead*nHead, T,  H, W]
        out = out.view(x.size(0), self.num_head, out.size(1), *x.size()[2:]).contiguous()
        out = out.view(*x.size())
        # [N, C, T, H, W]
        if self.use_gn:
            out = self.norm(out)

        out = residual + out
        return out


class MultiHeadNonLocal2d(_MultiHeadNonLocalNd):

    def __init__(self, inplanes, planes, downsample, num_head, distance, distance_delta, distance_mean, pos_embed_dim,
                 pos_feat_dim, pos_beta, use_saliency, saliency_alpha, use_gn, lr_mult, whiten_type, temp, nowd):
        print('MultiHeadNonLocal block, inplanes:{} planes:{} downsample:{} '
              'num_head:{} distance:{} distance_delta: {} distance_mean:{} '
              'pos_embed_dim:{} pos_feat_dim:{} pos_beta:{} '
              'use_saliency:{} saliency_alpha:{} '
              'use_gn:{} lr_mult:{} whiten_type:{} temp:{} nowd:{}'.format(inplanes, planes, downsample,
                                                                   num_head, distance, distance_delta, distance_mean,
                                                                   pos_embed_dim, pos_feat_dim, pos_beta,
                                                                   use_saliency, saliency_alpha, use_gn, lr_mult,
                                                                   whiten_type, temp, nowd))
        super(MultiHeadNonLocal2d, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample,
                                                  num_head=num_head, distance=distance, distance_delta=distance_delta,
                                                  distance_mean=distance_mean,
                                                  pos_embed_dim=pos_embed_dim, pos_feat_dim=pos_feat_dim,
                                                  pos_beta=pos_beta,
                                                  use_saliency=use_saliency, saliency_alpha=saliency_alpha,
                                                  use_gn=use_gn, lr_mult=lr_mult,
                                                  whiten_type=whiten_type, temp=temp, nowd=nowd)


class MultiHeadNonLocal3d(_MultiHeadNonLocalNd):

    def __init__(self, inplanes, planes, downsample, num_head, distance, distance_delta, distance_mean, pos_embed_dim,
                 pos_feat_dim, pos_beta, use_saliency, saliency_alpha, use_gn, lr_mult):
        super(MultiHeadNonLocal3d, self).__init__(dim=3, inplanes=inplanes, planes=planes, downsample=downsample,
                                                  num_head=num_head, distance=distance, distance_delta=distance_delta,
                                                  distance_mean=distance_mean,
                                                  pos_embed_dim=pos_embed_dim, pos_feat_dim=pos_feat_dim,
                                                  pos_beta=pos_beta,
                                                  use_saliency=use_saliency, saliency_alpha=saliency_alpha,
                                                  use_gn=use_gn, lr_mult=lr_mult)
