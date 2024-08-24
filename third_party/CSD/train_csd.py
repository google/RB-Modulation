#!/usr/bin/env python

import argparse
import json
import math
import os
import pathlib
import sys
import time
import datetime
import numpy as np
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from CSD import utils
from data.wikiart import WikiArtTrain
from data.laion import LAION, LAIONDedup
from CSD.loss_utils import ContrastiveTransformations, transforms_branch0, transforms_branch1, transforms_branch2
from CSD.model import CSD_CLIP
from CSD.losses import SupConLoss


def get_args_parser():

    parser = argparse.ArgumentParser('CSD', add_help=False)

    # Model
    parser.add_argument("-a","--arch",default='vit_base', type=str)

    # Data
    parser.add_argument('--train_set', default='wikiart',  # 'wikiart' or 'laion'
                    help='Wiki art data path')
    parser.add_argument('--train_path', required=True,
                        help='Wiki art data path')
    parser.add_argument('--train_anno_path',
                        default='-projects/diffusion_rep/data/laion_style_subset',
                        help='Annotation dir,  used only for LAION')
    parser.add_argument("--min_images_per_label", default=1, type=int, 
                        help="minimum images for a label (used only for laion)")
    parser.add_argument("--max_images_per_label", default=100000, type=int, 
                        help="minimum images for a label (used only for laion)")

    parser.add_argument('--eval_set', default='wikiart',  # 'domainnet' or 'wikiart'
                        help='Wiki art data path')
    parser.add_argument('--eval_path',required=True,
                        help='Path to query dataset.')
    parser.add_argument("--maxsize", default=512, type=int, 
                        help="maximum size of the val dataset to be used")

    # Optimization
    parser.add_argument( "--use_fp16", action="store_true",
                        help="use fp16")
    parser.add_argument( "--use_distributed_loss", action="store_true",
                        help="use distributed loss")
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help="""Maximal parameter gradient norm if using 
                        gradient clipping. Clipping with norm .3 ~ 1.0 can
                        help optimization for larger ViT architectures.
                        0 for disabling.""")
    parser.add_argument("--iters", default=100000, type=int,  # default: eval only
                        help="number of total iterations to run")
    parser.add_argument("-b", "--batch_size_per_gpu", default=64, type=int,
                        help="batch size per GPU (default: 64)")
    parser.add_argument("--lr", "--learning_rate", default=0.003, type=float,
                        help="learning rate", dest="lr",)
    parser.add_argument("--lr_bb", "--learning_rate_bb", default=0.0001, type=float,
                        help="learning rat for backbone", dest="lr_bb",)
    parser.add_argument("--wd", "--weight_decay", default=1e-4, type=float,
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--warmup_iters", default=30000, type=int,
                        help="Number of iterations for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
                        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')                   
    parser.add_argument('--freeze_last_layer', default=0, type=int,
                        help="""Number of iterations during which we keep the
                        output layer fixed. Typically doing so during 
                        first few iters helps training. Try increasing this 
                        value if the loss does not decrease.""")

    parser.add_argument('--content_proj_head', type=str, default='default')
    parser.add_argument('--lambda_s', default=1, type=float, help='Weighting on style loss')
    parser.add_argument('--lambda_c', default=0, type=float, help='Weighting on content loss')
    parser.add_argument('--lam_sup', default=5, type=float, help='Supervised style loss lambda')
    parser.add_argument('--temp', default=0.1, type=float, help='contrastive temperature')

    parser.add_argument('--clamp_content_loss', default=None, type=float, help='Clipping the content loss')
    parser.add_argument( "--non_adv_train", action="store_true",
                        help="dont train content adversarially, use neg of content loss")
    parser.add_argument('--eval_embed', type=str, default='head', help='which embeddings to use in evaluation')
    parser.add_argument('--style_loss_type', type=str, default='SupCon', help='which loss function for style loss computation')
    # Logging Params
    parser.add_argument('--output_dir', required=True, type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--print_freq', default=100, type=int, help='Print the logs every x iterations.')
    parser.add_argument('--saveckp_freq', default=5000, type=int, help='Save checkpoint every x iterations.')
    parser.add_argument('--eval_freq', default=5000, type=int, help='Eval the model every x iterations.')
    parser.add_argument('--eval_k', type=int, nargs='+', default=[1, 5, 100], help='eval map and recall at these k values.')

    # Misc
    parser.add_argument("--resume_if_available", action="store_true")
    parser.add_argument("--seed", default=42, type=int,
                        help="seed for initializing training. ")
    parser.add_argument("-j", "--workers", default=4, type=int,
                        help="number of data loading workers (default: 32)")
    parser.add_argument("--rank", default=-1, type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist_url", default="env://",
                        help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    return parser


def sample_infinite_data(loader, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    BIG_NUMBER = 9999999999999
    while True:
        # Randomize dataloader indices before every epoch:
        try:  # Only relevant for distributed sampler:
            shuffle_seed = torch.randint(0, BIG_NUMBER, (1,), generator=rng).item()
            loader.sampler.set_epoch(shuffle_seed)
        except AttributeError:
            pass
        for batch in loader:
            yield batch


def main():
    parser = argparse.ArgumentParser('CSD', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.non_adv_train:
        assert args.clamp_content_loss is not None, 'You have to clamp content loss in non-adv style of training'
    utils.init_distributed_mode(args)
    if args.seed is not None:
        utils.fix_random_seeds(args.seed)

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ======================= setup logging =======================
    if utils.is_main_process() and args.iters > 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # ======================= preparing data =======================
    if args.lambda_c < 1e-3:
        train_transforms = ContrastiveTransformations(transforms_branch1, transforms_branch1, transforms_branch2)
    else:
        train_transforms = ContrastiveTransformations(transforms_branch0, transforms_branch1, transforms_branch2)

    if args.train_set == 'wikiart':
        train_dataset = WikiArtTrain(
            args.train_path, 'database',
            transform=train_transforms)
    elif args.train_set == 'laion':
        train_dataset = LAION(
            args.train_path, args.train_anno_path,
            min_images_per_label=args.min_images_per_label,
            max_images_per_label=args.max_images_per_label,
            transform=train_transforms)
    elif args.train_set == 'laion_dedup':
        train_dataset = LAIONDedup(
            args.train_path, args.train_anno_path,
            transform=train_transforms)
    else:
        raise NotImplementedError

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size_per_gpu, drop_last=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    train_loader = sample_infinite_data(train_loader, args.seed)

    if args.eval_set == 'wikiart':
        vq_dataset = WikiArtTrain(
            args.eval_path, 'query', transform=transforms_branch0, maxsize=args.maxsize)
        vidx_dataset = WikiArtTrain(
            args.eval_path, 'database', transform=transforms_branch0, maxsize=8*args.maxsize)

    vq_loader = torch.utils.data.DataLoader(
        vq_dataset, batch_size=2*args.batch_size_per_gpu, drop_last=True,
        num_workers=min(args.workers, 2), pin_memory=True, shuffle=False)
    vidx_loader = torch.utils.data.DataLoader(
        vidx_dataset, batch_size=2*args.batch_size_per_gpu, drop_last=True,
        num_workers=min(args.workers, 2), pin_memory=True, shuffle=False)
    print(f"Data loaded: there are {len(train_dataset)} train images.")
    print(f"Data loaded: there are {len(vq_dataset)} query and {len(vidx_dataset)} index images.")

    # ======================= building model =======================
    model = CSD_CLIP(args.arch, args.content_proj_head) # TODO: projection dim into hyperparam
    model = model.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    print(f"Model built with {args.arch} network.")

    # ======================= setup loss and optimizers =======================
    loss_content = SupConLoss(temperature=args.temp) # TODO: Do we want 2 diff 
    loss_style = SupConLoss(temperature=args.temp)

    params_groups = utils.get_params_groups(model_without_ddp.backbone)
    # lr is set by scheduler
    opt_bb = torch.optim.SGD(
        params_groups, lr=0, momentum=0.9, weight_decay=args.weight_decay)

    if args.content_proj_head != 'default':
        opt_proj = torch.optim.SGD(
            [{'params': model_without_ddp.last_layer_style},
            {'params': model_without_ddp.last_layer_content.parameters()},],
            # [model_without_ddp.last_layer_style, *model_without_ddp.last_layer_content.parameters()],
            lr=0, momentum=0.9, weight_decay=0, # we do not apply weight decay
        )
    else:
        opt_proj = torch.optim.SGD(
            [model_without_ddp.last_layer_style, model_without_ddp.last_layer_content],
            lr=0, momentum=0.9, weight_decay=0, # we do not apply weight decay
        )

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ======================= init schedulers =======================
    if args.lr_scheduler_type =='cosine':
        lr_schedule_bb = utils.cosine_scheduler(
            args.lr_bb * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            min(args.min_lr, args.lr_bb),
            max(args.iters, 1), warmup_iters=min(args.warmup_iters, args.iters)
        )

        lr_schedule_proj = utils.cosine_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            min(args.min_lr, args.lr),
            max(args.iters, 1), warmup_iters=min(args.warmup_iters, args.iters)
        )
    elif args.lr_scheduler_type =='constant_with_warmup':
        lr_schedule_bb = utils.constant_with_warmup_scheduler(
            args.lr_bb * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            max(args.iters, 1), warmup_iters=min(args.warmup_iters, args.iters),
        )

        lr_schedule_proj = utils.constant_with_warmup_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            max(args.iters, 1), warmup_iters=min(args.warmup_iters, args.iters),
        )
    else:
        print('Using constant LR for training')
        lr_schedule_bb = utils.constant_with_warmup_scheduler(
            args.lr_bb * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            max(args.iters, 1), warmup_iters=0,
        )

        lr_schedule_proj = utils.constant_with_warmup_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            max(args.iters, 1), warmup_iters=0,
        )

    print(f"Loss, optimizer and schedulers ready.")

    # ======================= optionally resume training =======================
    to_restore = {"iter": 0}
    if args.resume_if_available:
        if not args.output_dir.endswith(".pth"):
            ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
        else:
            ckpt_path = args.output_dir
        utils.restart_from_checkpoint(
            ckpt_path,
            run_variables=to_restore,
            model_state_dict=model,
            opt_bb=opt_bb,
            opt_proj=opt_proj,
            fp16_scaler=fp16_scaler,
        )
        print(f"Start iter: {to_restore['iter']}")
    start_iter = to_restore["iter"]
    save_dict = None
    print("Running eval before training!")
    val_stats = evaluate(model, vq_loader, vidx_loader, fp16_scaler is not None, args.eval_k, args.eval_embed)
    if start_iter >= args.iters:
        print(f"Start iter {start_iter} >= Max iters {args.iters} training!")
        return

    start_time = time.time()
    print("Starting CSD training !")
    metric_logger = utils.MetricLogger(delimiter="  ", max_len=args.iters)
    header = 'Iter:'
    
    #TODO: Check if we need to set model to train mode
    model.eval()
    for iter, batch in enumerate(metric_logger.log_every(train_loader, 100, header)):
        # ======================= training =======================

        if iter < start_iter:
            continue

        if iter >= args.iters:
            break

        # update learning rates according to their schedule
        # it = len(train_loader) * epoch + it  # global training iteration
        p = float(iter) / args.iters

        for param_group in opt_bb.param_groups:
            param_group["lr"] = lr_schedule_bb[iter]

        for param_group in opt_proj.param_groups:
            param_group["lr"] = lr_schedule_proj[iter]
        if args.non_adv_train:
            alpha = None
        else:
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
        images, artists, *_ = batch
        if args.lambda_c < 1e-3:
            images = torch.cat([images[0],images[1]], dim=0)
        else:
            images = torch.cat(images, dim=0)

        # import torchvision
        # torchvision.utils.save_image(images,'./temp.png')
        images= images.cuda(non_blocking=True)
        artists = artists.cuda(non_blocking=True).float()

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            _ , content_output, style_output = model(images, alpha)

            # Normalize the output features for each image
            content_output = nn.functional.normalize(content_output, dim=1, p=2)
            style_output = nn.functional.normalize(style_output, dim=1, p=2)

            # Split the output features for each image and its views
            style_output  = utils.split_reshape(style_output, args.batch_size_per_gpu, [0, 1])
            content_output  = utils.split_reshape(content_output, args.batch_size_per_gpu, [0, -1])

            # Gather tensors from all GPUs
            if args.use_distributed_loss:
                style_output = torch.cat(utils.GatherLayer.apply(style_output), dim=0)
                content_output = torch.cat(utils.GatherLayer.apply(content_output), dim=0)

            # Compute content loss (SimCLR loss, doesn't use labels)
            loss_c = loss_content(content_output)
            if args.clamp_content_loss is not None:
                loss_c = loss_c.clamp(max = args.clamp_content_loss)
                if args.non_adv_train:
                    loss_c = -1 * loss_c

            # Compute style loss
            if args.use_distributed_loss:
                artists = torch.cat(utils.GatherLayer.apply(artists), dim=0)

            label_mask = artists @ artists.t()
            if args.style_loss_type == 'SimClr':
                loss_s_ssl = loss_style(style_output)
                loss_s_sup = torch.Tensor([0]).to(model.device)
            elif args.style_loss_type == 'OnlySup':
                loss_s_ssl = torch.Tensor([0]).to(model.device)
                loss_s_sup = loss_style(style_output[:, 0:1, :], mask=label_mask)
            else:
                loss_s_sup = loss_style(style_output[:, 0:1, :], mask=label_mask)
                loss_s_ssl = loss_style(style_output)

            loss_s = args.lam_sup*loss_s_sup + loss_s_ssl

        loss = args.lambda_c * loss_c + args.lambda_s * loss_s

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        opt_bb.zero_grad()
        opt_proj.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(iter, model, args.freeze_last_layer)
            opt_bb.step()
            opt_proj.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(opt_bb)  # unscale the gradients of optimizer's assigned params in-place
                fp16_scaler.unscale_(opt_proj)
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(iter, model, args.freeze_last_layer)
            fp16_scaler.step(opt_bb)
            fp16_scaler.step(opt_proj)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(content_loss=loss_c.item())
        metric_logger.update(style_loss=loss_s.item())
        metric_logger.update(style_loss_sup=loss_s_sup.item())
        metric_logger.update(style_loss_ssl=loss_s_ssl.item())
        metric_logger.update(lr_bb=opt_bb.param_groups[0]["lr"])
        # metric_logger.update(wd_bb=opt_bb.param_groups[0]["weight_decay"])
        metric_logger.update(lr_proj=opt_proj.param_groups[0]["lr"])
        # metric_logger.update(wd_proj=opt_proj.param_groups[0]["weight_decay"])

        # ============ writing logs ... ============
        save_dict = {
            'model_state_dict': model.state_dict(),
            'opt_bb': opt_bb.state_dict(),
            'opt_proj': opt_proj.state_dict(),
            'iter': iter+1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        if (iter+1) % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{iter+1:08}.pth'))

        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'iter': iter+1}

        if utils.is_main_process() and (iter+1) % args.print_freq == 0:
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Eval
        if (iter+1) % args.eval_freq==0:
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)

            val_stats = evaluate(model, vq_loader, vidx_loader, fp16_scaler is not None, args.eval_k, args.eval_embed)

    if args.iters > 0 and save_dict is not None:
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def evaluate(model, vq_loader, vidx_loader, use_fp16=False, eval_k=[1, 5, 100], eval_embed='head'):
    metric_logger = utils.MetricLogger(delimiter="  ")
    # Valid loader is the query set
    # Train loader is the search set
    use_cuda = True
    db_features = utils.extract_features(model, vidx_loader,use_cuda, use_fp16, eval_embed)
    q_features = utils.extract_features(model, vq_loader, use_cuda, use_fp16, eval_embed)

    # Aggregate style features across GPUs
    if utils.get_rank() != 0:
        return

    # Find the nearest neighbor indices for each query
    similarities = q_features @ db_features.T
    similarities = torch.argsort(similarities, dim=1, descending=True).cpu()

    # Map neighbor indices to labels (assuming one hot labels)
    q_labels = vq_loader.dataset.labels
    db_labels = vidx_loader.dataset.labels
    gts = q_labels @ db_labels.T
    #TODO: vectorize this
    preds = np.array([gts[i][similarities[i]] for i in range(len(gts))])

    # Compute metrics
    for topk in eval_k:
        mode_recall = utils.Metrics.get_recall_bin(copy.deepcopy(preds), topk)
        mode_mrr = utils.Metrics.get_mrr_bin(copy.deepcopy(preds), topk)
        mode_map = utils.Metrics.get_map_bin(copy.deepcopy(preds), topk)
        # print(f'Recall@{topk}: {mode_recall:.2f}, mAP@{topk}: {mode_map:.2f}')
        metric_logger.update(**{f'recall@{topk}': mode_recall, f'mAP@{topk}': mode_map, f'MRR@{topk}': mode_mrr})

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    main()
