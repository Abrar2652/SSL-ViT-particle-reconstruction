import torch


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats






import os

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def init_distributed(port=40112, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'distributed training not available {e}')

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads





import torch
from torch import optim

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


import math


class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd


import math

import torch

from logging import getLogger

logger = getLogger()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x


from typing import Dict

import torch
from torch import nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    """Barlow Twins model."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        lamb: float,
        batch_size: int,
    ):
        """Initializes the Barlow Twins model.

        Args:
            encoder (nn.Module): encoder to use.
            projector (nn.Module): projector to use.
            lamb (float): lambda parameter.
            batch_size (int): batch size.
            device (str): device to use.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.batch_size = batch_size
        self.lamb = lamb

        # Find out encoder len
        self.bn = nn.BatchNorm1d(projector[-1].out_features, affine=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass the Barlow Twins model."""
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # Empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.batch_size)

        # Sum the on-diagonal elements
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # Sum the off-diagonal elements
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamb * off_diag
        loss_normalized = (1 - self.lamb) / c.shape[0] * on_diag + self.lamb / (
            c.shape[0] * c.shape[0] - c.shape[0]
        ) * off_diag

        return {
            "loss": loss,
            "loss_norm": loss_normalized,
            "on_diag": on_diag,
            "off_diag": off_diag,
        }


import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, stride_list, planes_list, num_classes, num_channels=3, classification_head=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.resnet_blocks = []
        for block, stride, plane in zip(layer_list, stride_list, planes_list):
            self.resnet_blocks.append(self._make_layer(ResBlock, block, plane, stride))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if classification_head:
            self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        else:
            self.fc = nn.Identity()
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.resnet_blocks(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def Resnet(blocks, strides, planes, num_classes=2, channels=8, classification_head=False):
    assert len(blocks) == len(strides) == len(planes)
    return ResNet(Bottleneck, blocks, strides, planes, num_classes, channels, classification_head)
    
def ResNet50(blocks=[3,4,6,3], strides=[1,2,2,2], planes=[64,128,256,512],num_classes=2, channels=8, classification_head=False):
    return ResNet(Bottleneck, blocks, strides, planes, num_classes, channels, classification_head)



import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        ratio=(0.4, 0.6),
        input_size=(224, 224),
        patch_size=16,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.ratio = ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        ratio = self.ratio
        ratio = ratio[0] + torch.rand(1, generator=g).item() * (ratio[1] - ratio[0])
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - ratio))

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):

            m = torch.randperm(num_patches)
            collated_masks_enc.append([m[:num_keep]])
            collated_masks_pred.append([m[num_keep:]])

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()

def collate(data):
    
    jet = torch.cat([data[i][0] for i in range(len(data))], axis = 0) / 255
    meta = torch.cat([data[i][1] for i in range(len(data))], axis = 0)

    indexes = torch.randperm(jet.shape[0])
    jet = jet[indexes]
    meta = meta[indexes]
    meta = meta[:,0]
    return jet, meta


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        chunk_size=None,
        is_iris=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.chunk_size = chunk_size
        self.is_iris = is_iris

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        
        if self.is_iris:
            collated_batch = collate(batch)
            B = len(batch)*self.chunk_size
        else:
            collated_batch = torch.utils.data.default_collate(batch)
            B = len(batch)


        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class DefaultCollator(object):

    def __call__(self, batch):

        collated_batch = torch.utils.data.default_collate(batch)
        return collated_batch, None, None


import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=8, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=8, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=8,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=3,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}


import numpy as np
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, num_classes, use_batch_norm=False, use_hidden_layer=False, num_unfreeze_layers=0):
        super().__init__()
        self.encoder = pretrained_model

        # Freeze the encoder's parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        if num_unfreeze_layers > 0:
            for block in self.encoder.blocks[-num_unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        if use_hidden_layer:
            self.linear = nn.Sequential(
                nn.Linear(hidden_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,num_classes)
            )
        else:
            self.linear = nn.Linear(hidden_dim,num_classes)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = nn.Identity()


    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling

        x = self.batch_norm(x)
        return self.linear(x)
        
class AddLinear(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, use_batch_norm=False, use_hidden_layer=False):
        super().__init__()
        self.encoder = encoder

        if use_hidden_layer:
            self.linear = nn.Sequential(
                nn.Linear(hidden_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,num_classes)
            )
        else:
            self.linear = nn.Linear(hidden_dim,num_classes)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = nn.Identity()
    
    def forward(self,x):
        
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.batch_norm(x)

        return self.linear(x)

    

def resnet_50(num_classes, classification_head):
    return ResNet50(num_classes=num_classes,classification_head=classification_head)

def vit_model(num_classes, use_batch_norm, img_size, patch_size,model_name, use_hidden_layer):
    encoder = vit.__dict__[model_name](img_size=[img_size],patch_size=patch_size)
    embed_dim = encoder.embed_dim

    return AddLinear(encoder, embed_dim, num_classes, use_batch_norm, use_hidden_layer)



import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Subset

_GLOBAL_SEED = 0
logger = getLogger()


def make_gsoc_dataset(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
):
    
    dataset = GsocDataset3( root_path, preload_size=batch_size)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = ChunkedSampler(train_indices, chunk_size=batch_size, shuffle=True)
    val_sampler = ChunkedSampler(val_indices, chunk_size=batch_size, shuffle=False)

    train_data_loader = DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   pin_memory=pin_mem,
                                   collate_fn=collator,
                                   num_workers=num_workers)

    val_data_loader = DataLoader(dataset,
                                 batch_size=batch_size, 
                                 sampler=val_sampler, 
                                 pin_memory=pin_mem,
                                 collate_fn=collator, 
                                 num_workers=num_workers)
    
    logger.info('GSOC unsupervised data loaders created')

    return dataset, train_data_loader, val_data_loader


class ChunkedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = list(range(len(data_source)))
        self.shuffle = shuffle

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        np.random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

class GsocDataset3(torch.utils.data.Dataset):
    def __init__(self, h5_path, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['jet']
        #self.labels = self.h5_file['m0']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            #self.preloaded_labels = self.labels[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        #labels = self.preloaded_labels[local_idx]
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data)#, torch.from_numpy(labels)

    def __del__(self):
        self.h5_file.close()

        
def make_gsoc_dataset_iris(
    batch_size,
    split_size=None,
    chunk_size=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
):
    
    # Instantiate the dataset
    # mode can be 'train', 'test', or 'validation' depending on what you're doing
    train_dataset = Dataset4(file_path=root_path, mode='train', chunk_size=chunk_size)
    val_dataset = Dataset4(file_path=root_path, mode='validation', chunk_size=chunk_size)
    
    train_length = len(train_dataset)
    val_length = len(val_dataset)
    train_indices = list(range(int(train_length*split_size/100)))
    val_indices = list(range(int(val_length*50/100)))

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # Create the DataLoaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=True,  # Shuffle the data between epochs
        collate_fn=collator,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    # Create the DataLoader
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=True,  # Shuffle the data between epochs
        collate_fn=collator,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    logger.info('GSOC unsupervised data loaders created')

    return train_dataset, train_data_loader, val_data_loader

class Dataset4(torch.utils.data.Dataset):
    """Dataset Class"""

    def __init__(self, file_path,mode,chunk_size = 32):
        """
        Arguments:
            file_path (string): Path to the HDF5 file
            mode (string): "train", "test" or "validation" set to choose from.
            chunk_size: The chunk size to read the data from.
        """
        self.file_path = file_path
        self.mode = mode
        self.chunk_size = chunk_size

        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f[f"{self.mode}_jet"]) // self.chunk_size

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.file = h5py.File(self.file_path, 'r')

    def __getitem__(self, idx: int):

        if not hasattr(self, 'file'):
            self.open_hdf5()

        # Here idx is the chunk ID

        imgs = torch.tensor(self.file[f'{self.mode}_jet'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...].transpose(0,3,1,2))
        labels = torch.tensor(self.file[f'{self.mode}_meta'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...])
        return imgs, labels



import numpy as np

from logging import getLogger

import torch
import h5py
from torch.utils.data import DataLoader

_GLOBAL_SEED = 0
logger = getLogger()


def make_barlow_dataset(
    transforms_list,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
):
    
    dataset = GsocDataset3(root_path, preload_size=batch_size, transforms_list=transforms_list)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = ChunkedSampler(train_indices, chunk_size=batch_size, shuffle=True)
    val_sampler = ChunkedSampler(val_indices, chunk_size=batch_size, shuffle=False)

    train_data_loader = DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   pin_memory=pin_mem,
                                   #collate_fn=collator,
                                   num_workers=num_workers)

    val_data_loader = DataLoader(dataset,
                                 batch_size=batch_size, 
                                 sampler=val_sampler, 
                                 pin_memory=pin_mem,
                                 #collate_fn=collator, 
                                 num_workers=num_workers)
    
    logger.info('GSOC unsupervised data loaders created')

    return dataset, train_data_loader, val_data_loader


class ChunkedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = list(range(len(data_source)))
        self.shuffle = shuffle

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        np.random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

class GsocDataset3(torch.utils.data.Dataset):
    def __init__(self, h5_path, transforms_list=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms_list = transforms_list
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['jet']
        #self.labels = self.h5_file['m0']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            #self.preloaded_labels = self.labels[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data1 = torch.from_numpy(self.preloaded_data[local_idx])
        data2 = data1.clone()
        #labels = self.preloaded_labels[local_idx]
        if self.transforms_list:
            data1 = self.transforms_list(data1)
            data2 = self.transforms_list(data2)
        
        return data1, data2

    def __del__(self):
        self.h5_file.close()


import logging
import sys

import torch


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model(
    img_size,
    patch_size,
    pretrained_path=None,
    model_name=None,
    num_classes=1,
    use_batch_norm=False,
    use_hidden_layer=False,
    num_unfreeze_layers=0
):
    
    if pretrained_path is None:
        if 'resnet' in model_name:
            model = vit.__dict__[model_name](num_classes,classification_head=True)
        else:
            model = vit.__dict__['vit_model'](
                num_classes=num_classes, 
                use_batch_norm=use_batch_norm,
                img_size=img_size,
                patch_size=patch_size,
                model_name=model_name,
                use_hidden_layer=use_hidden_layer)

        return model
    
    else:
        encoder_ijepa = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size)
        embed_dim = encoder_ijepa.embed_dim
    
        checkpoint = torch.load(pretrained_path)
        encoder_ijepa.load_state_dict(checkpoint['target_encoder'])
        model = LinearProbe(encoder_ijepa, embed_dim, num_classes, use_batch_norm, use_hidden_layer, num_unfreeze_layers)
        return model





# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import numpy as np
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    color_jitter=1.0,
    color_distortion=False,
    horizontal_flip=False,
    vertical_flip=False,
    gaussian_blur=False,
    gaussian_blur_std=None,
    use_rotation=False
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if vertical_flip:
        transform_list += [transforms.RandomVerticalFlip()]
    if use_rotation:
        transform_list += [transforms.RandomRotation(30)]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [AddGaussianNoiseTorch(p=0.5, std=gaussian_blur_std)]

    return transforms.Compose(transform_list)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.prob=p
    
    def __call__(self, array):
        if np.random.rand() < self.prob:
            array = np.fliplr(array)
        return array

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.prob=p
    
    def __call__(self, array):
        if np.random.rand() < self.prob:
            array = np.flipud(array)
        return array

class AddGaussianNoise(object):
    def __init__(self, p=0.5, std=1.0, factor=0.1):
        self.prob= p
        assert len(std) == 8
        self.std = np.array(std).reshape(-1, 1, 1)
        self.factor = factor

    def __call__(self, tensor):
        if np.random.rand() < self.prob:
            return tensor
        
        noise = np.random.randn(*tensor.shape) * (self.std * self.factor)

        return tensor + noise

class AddGaussianNoiseTorch(object):
    def __init__(self, p=0.5, std=[1.0]*8, factor=0.1):
        self.prob = p
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.factor = factor

    def __call__(self, tensor):
        if torch.rand(1).item() >= self.prob:
            return tensor
        
        noise = torch.randn(tensor.size(), device=tensor.device) * (self.std * self.factor)
        return tensor + noise









import os

import logging
import sys
import yaml
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score

import os
import importlib.util
import sys


from torch.utils.data import Subset
from torch.utils.data import DataLoader

import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    model_name = args['meta']['model_name']
    pretrained_path = args['meta']['pretrained_path']
    use_batch_norm = args['meta']['use_batch_norm']
    use_hidden_layer = args['meta']['use_hidden_layer']
    num_unfreeze_layers = args['meta']['num_unfreeze_layers']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args['devices'][0])
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    patch_size = args['data']['patch_size']
    num_classes = args['data']['num_classes']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    img_size = args['data']['img_size']
    chunk_size = args['data']['chunk_size']
    
    # -- OPTIMIZATION
    num_epochs = args['optimization']['num_epochs']
    start_lr = args['optimization']['start_lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    folder = folder+'/Classification_'+str(args['fname'])
    if not os.path.exists(folder):
        os.makedirs(folder)
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_.csv')
    epoch_log_file = os.path.join(folder, f'{tag}_epoch.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                    ('%d', 'epoch'),
                    ('%d', 'itr'),
                    ('%.5f', 'train_loss'))
    
    # -- make csv_logger for end of epoch losses
    epoch_csv_logger = CSVLogger(epoch_log_file,
                      ('%d', 'epoch'),
                      ('%.5f', 'train_loss'),
                      ('%.5f', 'val_loss'),
                      ('%.5f', 'train_acc'),
                      ('%.5f', 'val_acc'),
                      ('%.5f', 'val_auc'))

    # -- init model
    model = init_model(
        img_size=img_size,
        patch_size=patch_size,
        pretrained_path=pretrained_path,
        model_name=model_name,
        num_classes=num_classes,
        use_batch_norm=use_batch_norm,
        use_hidden_layer=use_hidden_layer,
        num_unfreeze_layers=num_unfreeze_layers
    )

    # -- count parameter
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    logger.info('Number of parameters in the classifier: %d' % model_params)
    logger.info('Trainable params: %d' % trainable_params)

    # -- init dataset
    train_dataset = ProbingDataset(file_path=root_path, mode='train', chunk_size=chunk_size)
    val_dataset = ProbingDataset(file_path=root_path, mode='validation', chunk_size=chunk_size)

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    train_indices = list(range(int(train_length*95/100),train_length))
    val_indices = list(range(int(val_length*75/100),val_length))

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # -- init dataLoaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=True,  # Shuffle the data between epochs
        collate_fn=collate,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    # Create the DataLoader
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=False,  # Shuffle the data between epochs
        collate_fn=collate,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    logger.info('GSOC supervised data loaders created')

    # -- init optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    loss_function = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    model = model.to(device)

    def save_checkpoint(epoch):
        save_dict = {
            'model': model.state_dict()
        }

        torch.save(save_dict, save_path)

    start_epoch = 0
    # -- Early stopping parameters
    patience = 5  
    best_val_loss = float('inf') 
    epochs_no_improvement = 0 
    delta = 0.001

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, data in enumerate(train_data_loader):
            inputs, train_labels = data
            padding = (0, 1, 0, 1)
            inputs = torch.nn.functional.pad(inputs, padding, mode='constant', value=0)

            inputs = inputs.float().to(device)
            train_labels = train_labels.unsqueeze(1).to(device)
            #print('train labels: ', train_labels)

            # Forward pass
            outputs = model(inputs)
            #print('outputs: ', outputs)

            # Loss computation
            loss = loss_function(outputs, train_labels)
            total_loss += loss.item()

            # Calculate predictions for accuracy
            predicted_probabilities = torch.sigmoid(outputs)
            predicted_labels = (predicted_probabilities > 0.5).float()
            #print('pred labels: ', predicted_labels)
            correct_predictions += (predicted_labels == train_labels).sum().item()
            total_predictions += train_labels.size(0)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            csv_logger.log(epoch + 1, batch_idx, loss)

        epoch_loss = total_loss / len(train_data_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        print(f'Epoch {epoch+1}\nTrain loss: {epoch_loss:.4f}, Train accuracy: {epoch_accuracy:.2f}%')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for val_batch_idx, val_data in enumerate(val_data_loader):
                val_inputs, labels = val_data
                padding = (0, 1, 0, 1)
                val_inputs = torch.nn.functional.pad(val_inputs, padding, mode='constant', value=0)

                val_inputs = val_inputs.float().to(device)
                labels = labels.unsqueeze(1).to(device)

                # Forward pass
                logits = model(val_inputs)

                # Loss computation
                val_loss += loss_function(logits, labels).item()

                # Calculate predictions for accuracy
                val_predicted_probabilities = torch.sigmoid(logits)
                val_predicted_labels = (val_predicted_probabilities > 0.5).float()
                val_correct_predictions += (val_predicted_labels == labels).sum().item()
                val_total_predictions += labels.size(0)

                # Store labels and logits for auc
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())

        val_epoch_loss = val_loss / len(val_data_loader)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions * 100
        auc_score = roc_auc_score(val_labels, val_preds)
        print(f'Validation loss: {val_epoch_loss:.4f}, Validation accuracy: {val_epoch_accuracy:.2f}%, Val AUC score:{auc_score:.4f}')

        epoch_csv_logger.log(epoch + 1, epoch_loss, val_epoch_loss, epoch_accuracy, val_epoch_accuracy, auc_score)

        if best_val_loss - val_epoch_loss > delta:
            best_val_loss = val_epoch_loss
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    save_checkpoint(epoch)


if __name__ == "__main__":
    args = parser.parse_args()

    # -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
  
    
    params['devices'] = args.devices
    params['fname'] = args.fname

    main(args=params)
