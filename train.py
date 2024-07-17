from engine import train_one_epoch, evaluate
import torch
from utils import utils
from models.model import build_model
import argparse
import numpy as np
from dataset.Dota import build
from pathlib import Path
import utils.misc as misc
import random
import os
import dataset.samplers as samplers
import json

def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone"], type=str, nargs="+"
    )

    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)

    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--sgd", action="store_true")


    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned", "sine_unnorm"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=1, type=int, help="number of feature levels"
    )
    # SAM backbone
    parser.add_argument(
        "--pretrained_backbone_path",
        default="../weights/sam_vit_b_01ec64.pth",
        type=str,
    )
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    # upsample backbone output features
    parser.add_argument(
        "--upsample_backbone_output",
        action="store_true",
        help="If true, we upsample the backbone output feature to the target stride"
    )
    parser.add_argument(
        "--upsample_stride",
        default=16,
        type=int,
        help="Target stride for upsampling backbone output feature"
    )

    

    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--not_auto_resume", action="store_false", dest="auto_resume")

    # weight decay mult
    parser.add_argument(
        "--wd_norm_names",
        default=["norm", "bias", "rpb_mlp", "cpb_mlp", "logit_scale", "relative_position_bias_table",
                 "level_embed", "reference_points", "sampling_offsets", "rel_pos"],
        type=str,
        nargs="+"
    )
    parser.add_argument("--wd_norm_mult", default=1.0, type=float)
    parser.add_argument("--use_layerwise_decay", action="store_true", default=False)
    parser.add_argument("--lr_decay_rate", default=1.0, type=float)

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
        default=True
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="/home/bibahaduri/dota_dataset/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="./output", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    # * eval technologies
    parser.add_argument("--eval", action="store_true")

    # * training technologies
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--use_checkpoint", default=False, action="store_true")

    # * logging technologies
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_name", type=str)
    return parser









def main(args):
    # train on the GPU or on the CPU, if a GPU is not available


    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = build('train', args)
    dataset_test = build('val', args)

    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset)
        sampler_val = samplers.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # get the model using our helper function
    model = build_model(num_classes=2, args=args)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = misc.get_param_dict(model, args, use_layerwise_decay=args.use_layerwise_decay)##[p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # and a learning rate scheduler
    epoch_iter = len(data_loader)
    if args.warmup:
        lambda0 = lambda cur_iter: cur_iter / args.warmup if cur_iter < args.warmup else (0.1 if cur_iter > args.lr_drop * epoch_iter else 1)
    else:
        lambda0 = lambda cur_iter: 0.1 if cur_iter > args.lr_drop * epoch_iter else 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=3,
    #     gamma=0.1
    # )


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    output_dir = Path(args.output_dir)


    if args.auto_resume:
        resume_from = misc.find_latest_checkpoint(output_dir)
        if resume_from is not None:
            print(f'Use autoresume, overwrite args.resume with {resume_from}')
            args.resume = resume_from
        else:
            print(f'Use autoresume, but can not find checkpoint in {output_dir}')
    if args.resume and os.path.exists(args.resume):
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))

        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))

        if (
            not args.eval
            and args.auto_resume
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy

            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]

            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print(
                # For LambdaLR, the lambda funcs are not been stored in state_dict, see
                # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR.state_dict
                "Warning: lr scheduler has been resumed from checkpoint, but the lambda funcs are not been stored in state_dict. \n"
                "So the new lr schedule would override the resumed lr schedule."
            )
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1


    if args.eval:
        test_stats, evalutor = evaluate(model, data_loader_test, device=device)
        print(test_stats)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        train_stats = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 5 epochs
            #checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                # if args.use_fp16:
                #     save_dict["scaler"] = scaler.state_dict()
                utils.save_on_master(
                    save_dict,
                    checkpoint_path,
                )
        # evaluate on the test dataset
        test_stats, evalutor = evaluate(model, data_loader_test, device=device)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            #"n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)