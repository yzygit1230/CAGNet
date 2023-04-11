import os
from copy import deepcopy

from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CAGNet import CAGNet
from models.CAGNet import GAmodule
from collections import OrderedDict
from util.common import ScaleInOutput
import ml_collections

def get_CAGNet_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64
    config.n_classes = 2
    return config

config = get_CAGNet_config()
CAG = CAGNet(config, n_channels=3, img_size=512)

class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = 64
        self.auxiliary_head = False
        self.model = CAG
        self.neck = GAmodule(self.inplanes)
        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)

    def forward(self, xa, xb, tta=False):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        out = self.model(xa, xb)
        return out

    def _init_weight(self, pretrain=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))


class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=448):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        print("-"*50+"\n--Ensamble method: {}".format(method))
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb, tta=False):
        xa, xb = self.scale.scale_input((xa, xb))
        out1 = 0
        cd_pred = None

        for i, model in enumerate(self.models_list):
            outs = model(xa, xb, tta)
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                _, cd_pred = torch.max(out1, 1)

        return cd_pred

class ModelEMA:
    def __init__(self, model, decay=0.96):
        self.shadow1 = deepcopy(model.module if self.is_parallel(model) else model).eval()
        self.decay = decay
        for p in self.shadow1.parameters():
            p.requires_grad_(False)
        self.shadow2 = deepcopy(self.shadow1)
        self.shadow3 = deepcopy(self.shadow1)
        self.update_count = 0

    def update(self, model):
        with torch.no_grad():
            msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
            for k, v in self.shadow1.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1. - self.decay) * msd[k].detach()
            for k, v in self.shadow2.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.95
                    v += (1. - 0.95) * msd[k].detach()
            for k, v in self.shadow3.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.94
                    v += (1. - 0.94) * msd[k].detach()
        self.update_count += 1

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelSWA:
    def __init__(self, total_epoch=300):
        self.update_count = 0
        self.epoch_threshold = int(total_epoch * 0.8)
        self.swa_model = None

    def update(self, model):
        if self.update_count >= self.epoch_threshold:
            with torch.no_grad():
                if self.swa_model is None:
                    self.swa_model = deepcopy(model.module) if self.is_parallel(model) else deepcopy(model)
                else:
                    msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
                    for k, v in self.swa_model.state_dict().items():
                        if v.dtype.is_floating_point:
                            v *= (self.update_count - self.epoch_threshold)
                            v += msd[k].detach()
                            v /= (self.update_count - self.epoch_threshold + 1)
        self.update_count += 1

    def save(self, swa_ckp_dir_path):
        if self.update_count >= self.epoch_threshold:
            swa_file_path = os.path.join(swa_ckp_dir_path, "swa_{}_{}.pt".format(
                self.update_count-1, self.update_count-self.epoch_threshold))
            torch.save(self.swa_model, swa_file_path)

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
