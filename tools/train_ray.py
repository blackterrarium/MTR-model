# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Ray Training Implementation
# Written for Ray distributed training
# All Rights Reserved

import _init_path
import argparse
import datetime
import os
import tempfile
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train import Checkpoint

from mtr.datasets import build_dataloader
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.utils import common_utils
from mtr.models import model as model_utils

def setup_s3_credentials():
    """Setup S3 credentials for MinIO access"""
    # Set environment variables for S3/MinIO access
    # These will be used by PyArrow/fsspec for S3 access
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MINIO_ACCESS_KEY', '')
    if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MINIO_SECRET_KEY', '')
    if 'AWS_ENDPOINT_URL' not in os.environ:
        os.environ['AWS_ENDPOINT_URL'] = os.getenv('MINIO_ENDPOINT', '')
    if 'AWS_REGION' not in os.environ:
        os.environ['AWS_REGION'] = os.getenv('AWS_REGION', 'us-east-1')
    
    # For S3-compatible storage (MinIO), we need to set additional config
    os.environ['AWS_S3_ALLOW_UNSAFE_RENAME'] = 'true'


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser for Ray training')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')

    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    
    # Ray-specific arguments
    parser.add_argument('--num_workers', type=int, default=2, help='number of Ray workers')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use GPU for training')
    parser.add_argument('--storage_path', type=str, default='s3://mtr-training', help='Ray storage path (s3://bucket or local path)')
    parser.add_argument('--experiment_name', type=str, default='mtr_ray_training', help='Ray experiment name')
    
    # S3/MinIO configuration
    parser.add_argument('--s3_data_root', type=str, default='s3://mtr-data', help='S3 path for data')
    parser.add_argument('--minio_endpoint', type=str, default=None, help='MinIO endpoint URL')
    parser.add_argument('--minio_access_key', type=str, default=None, help='MinIO access key')
    parser.add_argument('--minio_secret_key', type=str, default=None, help='MinIO secret key')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def build_optimizer(model, opt_cfg):
    if opt_cfg.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters()],
            lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
        )
    elif opt_cfg.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0))
    else:
        assert False

    return optimizer


def build_scheduler(optimizer, dataloader, opt_cfg, total_epochs, total_iters_each_epoch, last_epoch):
    decay_steps = [x * total_iters_each_epoch for x in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * opt_cfg.LR_DECAY
        return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

    if opt_cfg.get('SCHEDULER', None) == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2 * len(dataloader),
            T_mult=1,
            eta_min=max(1e-2 * opt_cfg.LR, 1e-6),
            last_epoch=-1,
        )
    elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
        scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
        total_iters = total_iters_each_epoch * total_epochs
        scheduler = lr_sched.LinearLR(optimizer, start_factor=1.0, end_factor=opt_cfg.LR_CLIP / opt_cfg.LR, total_iters=total_iters, last_epoch=last_epoch)
    else:
        scheduler = None

    return scheduler


def train_func(config):
    """
    Ray training function that runs on each worker
    """
    # Setup S3 credentials on each worker
    setup_s3_credentials()
    
    # Get configuration from train_loop_config
    args = config['args']
    cfg = config['cfg']
    
    # Set random seed if requested
    if args.fix_random_seed:
        common_utils.set_random_seed(666)
    
    # Get Ray context information
    world_size = ray.train.get_context().get_world_size()
    rank = ray.train.get_context().get_world_rank()
    local_rank = ray.train.get_context().get_local_rank()
    
    # Adjust batch size per worker
    if args.batch_size is None:
        worker_batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        worker_batch_size = args.batch_size // world_size
    
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    
    # Update cfg with S3 data paths
    if args.s3_data_root:
        cfg.DATA_CONFIG.DATA_ROOT = args.s3_data_root
    
    # Create logger (only on rank 0)
    logger = None
    if rank == 0:
        log_file = f'log_train_ray_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
        logger = common_utils.create_logger(log_file, rank=rank)
        logger.info('**********************Start Ray Training**********************')
        logger.info(f'Total workers: {world_size}')
        logger.info(f'Worker batch size: {worker_batch_size}')
        logger.info(f'Global batch size: {worker_batch_size * world_size}')
        for key, val in vars(args).items():
            logger.info(f'{key:16} {val}')
        log_config_to_file(cfg, logger=logger)
    
    # Build datasets and dataloaders
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=worker_batch_size,
        dist=True,  # Always use distributed for Ray
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
    )
    
    # Prepare dataloader for Ray Train
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    
    # Build model
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    # Convert sync batch norm if needed
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Prepare model for Ray Train (handles device placement and DDP wrapping)
    model = ray.train.torch.prepare_model(model)
    
    # Build optimizer
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    
    # Load checkpoint if specified
    start_epoch = 0
    start_iter = 0
    
    # Check for Ray Train checkpoint first
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(checkpoint_data["model_state_dict"])
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                start_epoch = checkpoint_data["epoch"]
                start_iter = checkpoint_data["iteration"]
                if logger:
                    logger.info(f"Loaded Ray checkpoint from epoch {start_epoch}")
    
    # Load pretrained model if specified
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)
    
    # Build scheduler
    scheduler = build_scheduler(
        optimizer, train_loader, cfg.OPTIMIZATION, total_epochs=args.epochs,
        total_iters_each_epoch=len(train_loader), last_epoch=start_epoch-1
    )
    
    # Training loop
    accumulated_iter = start_iter
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        elif ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Update scheduler
            if scheduler is not None:
                try:
                    scheduler.step(accumulated_iter)
                except:
                    scheduler.step()
            
            # Get current learning rate
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            
            # Forward pass
            optimizer.zero_grad()
            loss, tb_dict, disp_dict = model(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            
            # Optimizer step
            optimizer.step()
            
            accumulated_iter += 1
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if rank == 0 and (batch_idx % args.logger_iter_interval == 0 or batch_idx == len(train_loader) - 1):
                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                disp_str += f', lr={cur_lr:.6f}'
                batch_size = batch.get('batch_size', worker_batch_size)
                if logger:
                    logger.info(f'epoch: {epoch}/{args.epochs}, iter: {batch_idx}/{len(train_loader)}, '
                              f'batch_size: {batch_size}, accumulated_iter: {accumulated_iter}, {disp_str}')
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Prepare metrics for reporting
        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": cur_lr,
            "accumulated_iter": accumulated_iter
        }
        
        # Save checkpoint
        checkpoint_dict = None
        if (epoch + 1) % args.ckpt_save_interval == 0 or epoch + 1 == args.epochs:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint_path = os.path.join(temp_checkpoint_dir, "model_checkpoint.pth")
                
                # Get the underlying model (unwrap from DDP if necessary)
                model_to_save = model.module if hasattr(model, 'module') else model
                
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "iteration": accumulated_iter,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                checkpoint_dict = Checkpoint.from_directory(temp_checkpoint_dir)
                
                if logger:
                    logger.info(f"Saved checkpoint at epoch {epoch + 1}")
        
        # Report metrics and checkpoint to Ray Train
        ray.train.report(metrics=metrics, checkpoint=checkpoint_dict)


def main():
    args, cfg = parse_config()
    
    # Setup S3 credentials
    setup_s3_credentials()
    
    # Override environment variables if provided
    if args.minio_endpoint:
        os.environ['MINIO_ENDPOINT'] = args.minio_endpoint
        os.environ['AWS_ENDPOINT_URL'] = args.minio_endpoint
    if args.minio_access_key:
        os.environ['MINIO_ACCESS_KEY'] = args.minio_access_key
        os.environ['AWS_ACCESS_KEY_ID'] = args.minio_access_key
    if args.minio_secret_key:
        os.environ['MINIO_SECRET_KEY'] = args.minio_secret_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.minio_secret_key
    
    print("**********************Starting Ray Training Setup**********************")
    print(f"Ray storage path: {args.storage_path}")
    print(f"S3 data root: {args.s3_data_root}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Use GPU: {args.use_gpu}")
    
    # Configure Ray scaling
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        resources_per_worker={"CPU": 1, "GPU": 1 if args.use_gpu else 0}
    )
    
    # Configure storage and run settings
    run_config = RunConfig(
        storage_path=args.storage_path,
        name=f"{args.experiment_name}_{args.extra_tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        stop={"training_iteration": args.epochs} if args.epochs else None,
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=args.max_ckpt_save_num,
            checkpoint_score_attribute="train_loss",
            checkpoint_score_order="min"
        )
    )
    
    # Create trainer configuration
    train_loop_config = {
        "args": args,
        "cfg": cfg,
    }
    
    # Create and run trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    print("**********************Starting Ray Training**********************")
    result = trainer.fit()
    
    print("**********************Training Completed**********************")
    print(f"Best checkpoint: {result.checkpoint}")
    print(f"Final metrics: {result.metrics}")
    print(f"Training logs path: {result.path}")


if __name__ == '__main__':
    main()
