import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_BN(nn.Module):
    def __init__(self, nb_in, nb_out, ksize=1, pad=0, no_bn=False):
        super(Conv_BN, self).__init__()
        self.no_bn = no_bn
        self.conv = nn.Conv1d(nb_in, nb_out, kernel_size=ksize, padding=pad)
        if not no_bn:
            self.bn = nn.BatchNorm1d(nb_out)

    def forward(self, x):
        if self.no_bn:
            return self.conv(x)
        else:
            return F.relu(self.bn(self.conv(x)))


class Conv_Module(nn.Module):
    def __init__(self, nb_in, nb_out, inter_list=[], no_act_last=False):
        super(Conv_Module, self).__init__()
        self.nb_layers = len(inter_list) + 1
        layers = []
        if len(inter_list) == 0:
            layers.append(Conv_BN(nb_in, nb_out, no_bn=no_act_last))
        else:
            layers.append(Conv_BN(nb_in, inter_list[0]))
            for nin, nout in zip(inter_list[:-1], inter_list[1:]):
                layers.append(Conv_BN(nin, nout))
            layers.append(Conv_BN(inter_list[-1], nb_out, no_bn=no_act_last))
        self.layers = nn.ModuleList(layers)

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class Encoder(nn.Module):
    def __init__(self, nb_inputs, channel_list, ksize_list, pad_list=[]):
        super(Encoder, self).__init__()
        self.nb_layers = len(channel_list)
        channel_list = [nb_inputs] + channel_list
        if len(pad_list) == 0:
            pad_list = [0 for _ in range(len(ksize_list))]
        layers = []
        for nb_in, nb_out, ksize, pad in zip(channel_list[:-1], channel_list[1:], ksize_list, pad_list):
            layers.append(Conv_BN(nb_in, nb_out, ksize, pad))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = torch.transpose(x, 1, 2)
        for layer in self.layers:
            h = layer(h)
        return h


class Decoder(nn.Module):
    def __init__(self, nb_inputs, channel_list, ksize_list, no_act_last=False):
        super(Decoder, self).__init__()
        self.nb_layers = len(channel_list)
        self.no_act_last = no_act_last
        channel_list = channel_list + [nb_inputs]
        layers = []
        for idx, (nb_in, nb_out, ksize) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list[::-1])):
            layers.append(nn.ConvTranspose1d(nb_in, nb_out, kernel_size=ksize))
            if not (no_act_last and idx == self.nb_layers - 1):
                layers.append(nn.BatchNorm1d(nb_out))
        self.layers = nn.ModuleList(layers)

    def forward(self, h):
        for idx in range(self.nb_layers):
            layer_idx = idx * 2
            if self.no_act_last and idx == self.nb_layers - 1:
                h = self.layers[layer_idx](h)
            else:
                h = F.relu(self.layers[layer_idx + 1](self.layers[layer_idx](h)))
        return h