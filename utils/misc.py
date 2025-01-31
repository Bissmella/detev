# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import copy
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        ## type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ["LOCAL_SIZE"])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.dist_url = "env://"
        os.environ["LOCAL_SIZE"] = str(torch.cuda.device_count())
    elif "SLURM_PROCID" in os.environ:
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            "scontrol show hostname {} | head -n1".format(node_list)
        )
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
        os.environ["LOCAL_SIZE"] = str(num_gpus)
        args.dist_url = "env://"
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def find_latest_checkpoint(path, ext="pth"):
    import glob
    import os.path as osp

    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"checkpoint.{ext}")):
        return osp.join(path, f"checkpoint.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    checkpoints = [ckpt for ckpt in checkpoints if osp.basename(ckpt) != 'eval.pth']
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).lstrip("checkpoint").split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_swin_layer_id(var_name, backbone_type):
    maps = {
        "tiny": dict(num_max_layer=12, layers_per_stage=[2, 2, 6, 2]),
        "small": dict(num_max_layer=24, layers_per_stage=[2, 2, 18, 2]),
        "base": dict(num_max_layer=24, layers_per_stage=[2, 2, 18, 2]),
        "large": dict(num_max_layer=24, layers_per_stage=[2, 2, 18, 2]),
    }
    map_type = None
    for k in maps.keys():
        if k in backbone_type:
            map_type = k
            break
    assert map_type is not None, f"Unsupported backbone type {backbone_type}"

    # hack for UpSampleWrapper
    if var_name.startswith('backbone.0.net.'):
        num_max_layer = maps[map_type]["num_max_layer"]
        layers_per_stage = maps[map_type]["layers_per_stage"]
        if var_name.startswith("backbone.0.net.body.patch_embed"):
            layer_id = 0
        elif var_name.startswith("backbone.0.net.body.layers"):
            if var_name.split('.')[6] == "blocks":
                stage_id = int(var_name.split('.')[5])
                layer_id = int(var_name.split('.')[7]) + sum(layers_per_stage[:stage_id])
                layer_id = layer_id + 1
            elif var_name.split('.')[6] == "downsample":
                stage_id = int(var_name.split('.')[5])
                layer_id = sum(layers_per_stage[:stage_id+1])
                layer_id = layer_id
        elif var_name.startswith("backbone.0.net.body.norm"):
            layer_id = num_max_layer + 1
        else:
            layer_id = num_max_layer + 1
        return num_max_layer + 1 - layer_id

    num_max_layer = maps[map_type]["num_max_layer"]
    layers_per_stage = maps[map_type]["layers_per_stage"]
    if var_name.startswith("backbone.0.body.patch_embed"):
        layer_id = 0
    elif var_name.startswith("backbone.0.body.layers"):
        if var_name.split('.')[5] == "blocks":
            stage_id = int(var_name.split('.')[4])
            layer_id = int(var_name.split('.')[6]) + sum(layers_per_stage[:stage_id])
            layer_id = layer_id + 1
        elif var_name.split('.')[5] == "downsample":
            stage_id = int(var_name.split('.')[4])
            layer_id = sum(layers_per_stage[:stage_id+1])
            layer_id = layer_id
    elif var_name.startswith("backbone.0.body.norm"):
        layer_id = num_max_layer + 1
    else:
        layer_id = num_max_layer + 1
    return num_max_layer + 1 - layer_id


def get_layerwise_param_dict(model, args, return_name=False):

    parameter_groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names):
            if match_name_keywords(n, args.wd_norm_names):
                group_name = "wd_mult"
                weight_decay = args.weight_decay * args.wd_norm_mult
            else:
                group_name = "wd"
                weight_decay = args.weight_decay
            if 'swin' in args.backbone:
                layer_id = get_swin_layer_id(n, args.backbone)
            else:
                raise NotImplementedError
            group_name = f"layer_{layer_id}_{group_name}"

            if group_name not in parameter_groups:
                scale = args.lr_decay_rate ** layer_id

                parameter_groups[group_name] = {
                    "params": [],
                    # "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * args.lr,
                    "weight_decay": weight_decay,
                }
            parameter_groups[group_name]["params"].append(n if return_name else p)
            # parameter_groups[group_name]["param_names"].append(name)
        elif not match_name_keywords(n, args.lr_backbone_names) and match_name_keywords(n, args.lr_linear_proj_names):
            if match_name_keywords(n, args.wd_norm_names):
                group_name = "wd_mult"
                weight_decay = args.weight_decay * args.wd_norm_mult
            else:
                group_name = "wd"
                weight_decay = args.weight_decay
            group_name = f"head_linear_proj_{group_name}"

            if group_name not in parameter_groups:
                scale = args.lr_linear_proj_mult

                parameter_groups[group_name] = {
                    "params": [],
                    # "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * args.lr,
                    "weight_decay": weight_decay,
                }
            parameter_groups[group_name]["params"].append(n if return_name else p)
        elif not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names):
            if match_name_keywords(n, args.wd_norm_names):
                group_name = "wd_mult"
                weight_decay = args.weight_decay * args.wd_norm_mult
            else:
                group_name = "wd"
                weight_decay = args.weight_decay
            group_name = f"head_{group_name}"

            if group_name not in parameter_groups:
                scale = 1.0

                parameter_groups[group_name] = {
                    "params": [],
                    # "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * args.lr,
                    "weight_decay": weight_decay,
                }
            parameter_groups[group_name]["params"].append(n if return_name else p)
        else:
            raise ValueError

    param_dicts = list(parameter_groups.values())

    return param_dicts


def get_param_dict(model, args, return_name=False, use_layerwise_decay=False):
    # sanity check: a variable could not match backbone_names and linear_proj_names at the same time
    for n, p in model.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and match_name_keywords(n, args.lr_linear_proj_names):
            raise ValueError

    if use_layerwise_decay:
        return get_layerwise_param_dict(model, args, return_name)

    param_dicts = [
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
    ]
    return param_dicts


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: torch.Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)