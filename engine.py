import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils.utils as utils
from dataset.coco_eval import CocoEvaluator
from dataset.coco_utils import get_coco_api_from_dataset
from utils.misc import ImageList
from typing import Tuple, List


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.stack([image.to(device) for image in images])##list(image.to(device) for image in images)   ##
        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list = [(t['size'][0].item(), t['size'][0].item()) for t in targets] ##: List[Tuple[int, int]] = []

        # for image_size in image_sizes:
        #     torch._assert(
        #         len(image_size) == 2,
        #         f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
        #     )
        #     image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict, _ = model(image_list, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}##metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack([image.to(device) for image in images])##list(image.to(device) for image in images)   ##
        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list = [(t['size'][0].item(), t['size'][0].item()) for t in targets] ##: List[Tuple[int, int]] = []

        # for image_size in image_sizes:
        #     torch._assert(
        #         len(image_size) == 2,
        #         f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
        #     )
        #     image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        loss_dict, outputs = model(image_list, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        if losses:
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if losses_reduced:
                loss_value = losses_reduced.item()

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        if losses:
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    loss_details = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    loss_details['coco_eval_bbox'] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    loss_details['coco_eval_masks'] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return loss_details, coco_evaluator