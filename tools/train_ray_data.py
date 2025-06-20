# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Ray Training Implementation with Ray Data Integration
# Written for Ray distributed training with optimized data loading
# All Rights Reserved

import _init_path
import argparse
import datetime
import os
import tempfile
from pathlib import Path
import math
import pickle
from typing import Dict, Any, Iterator

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

import ray
import ray.train
import ray.data
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train import Checkpoint

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


def build_scheduler(optimizer, opt_cfg, total_epochs, total_iters_each_epoch, last_epoch):
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
            T_0=2 * total_iters_each_epoch,
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


def create_ray_dataset_from_scenario_files(data_root: str, info_file: str, dataset_cfg, training: bool = True):
    """
    Create Ray Dataset directly from scenario files for better distributed loading
    """
    import numpy as np
    
    # Load info file to get scenario list
    info_path = os.path.join(data_root, info_file)
    if info_path.startswith('s3://'):
        # For S3 files, we need to download or read directly
        import boto3
        import io
        
        # Parse S3 path
        s3_parts = info_path.replace('s3://', '').split('/')
        bucket = s3_parts[0]
        key = '/'.join(s3_parts[1:])
        
        # Download pickle file
        s3_client = boto3.client('s3',
                               endpoint_url=os.environ.get('AWS_ENDPOINT_URL'),
                               aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                               aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                               region_name=os.environ.get('AWS_REGION', 'us-east-1'))
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        infos = pickle.loads(response['Body'].read())
    else:
        # Local file
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
    
    # Apply sampling and filtering
    sample_interval = dataset_cfg.SAMPLE_INTERVAL.get('train' if training else 'test', 1)
    infos = infos[::sample_interval]
    
    # Filter by object type if specified
    if hasattr(dataset_cfg, 'INFO_FILTER_DICT') and 'filter_info_by_object_type' in dataset_cfg.INFO_FILTER_DICT:
        valid_object_types = dataset_cfg.INFO_FILTER_DICT['filter_info_by_object_type']
        filtered_infos = []
        for cur_info in infos:
            num_interested_agents = len(cur_info['tracks_to_predict']['track_index'])
            if num_interested_agents == 0:
                continue
            
            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)
            
            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue
                
            # Filter the tracks_to_predict
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])
            
            filtered_infos.append(cur_info)
        infos = filtered_infos
    
    print(f"Created dataset with {len(infos)} scenarios after filtering")
    
    # Create list of scenario file paths
    split_dir = dataset_cfg.SPLIT_DIR.get('train' if training else 'test', 'processed_scenarios_training')
    scenario_paths = []
    for info in infos:
        scenario_id = info['scenario_id']
        scenario_path = os.path.join(data_root, split_dir, f'sample_{scenario_id}.pkl')
        scenario_paths.append({
            'scenario_path': scenario_path,
            'scenario_id': scenario_id,
            'info': info
        })
    
    # Create Ray Dataset from scenario paths
    ray_dataset = ray.data.from_items(scenario_paths)
    
    return ray_dataset


def load_and_process_scenario(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ray Data map function to load and process scenario data
    """
    import torch
    import numpy as np
    import pickle
    import boto3
    import io
    import os
    from mtr.utils import common_utils
    
    processed_batch = []
    
    for item in batch:
        scenario_path = item['scenario_path']
        scenario_id = item['scenario_id']
        info_data = item['info']
        
        # Load scenario file
        try:
            if scenario_path.startswith('s3://'):
                # Load from S3
                s3_parts = scenario_path.replace('s3://', '').split('/')
                bucket = s3_parts[0]
                key = '/'.join(s3_parts[1:])
                
                s3_client = boto3.client('s3',
                                       endpoint_url=os.environ.get('AWS_ENDPOINT_URL'),
                                       aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                                       aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                                       region_name=os.environ.get('AWS_REGION', 'us-east-1'))
                
                response = s3_client.get_object(Bucket=bucket, Key=key)
                info = pickle.loads(response['Body'].read())
            else:
                # Load from local file
                with open(scenario_path, 'rb') as f:
                    info = pickle.load(f)
        except Exception as e:
            print(f"Failed to load scenario {scenario_path}: {e}")
            continue
        
        # Process the scenario using MTR dataset logic
        # This is a simplified version - you may need to adapt based on your specific processing needs
        
        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)
        
        track_infos = info['track_infos']
        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]
        
        # Simplified processing - for full implementation, you'd need to port the entire
        # create_scene_level_data method from WaymoDataset
        
        # Create a basic return dictionary
        ret_dict = {
            'scenario_id': np.array([scenario_id]),
            'obj_trajs': obj_trajs_past,  # Simplified
            'obj_trajs_mask': np.ones_like(obj_trajs_past[:, :, 0]),  # Simplified
            'track_index_to_predict': track_index_to_predict,
            'obj_types': obj_types,
            'obj_ids': obj_ids,
            # Add other required fields as needed
        }
        
        processed_batch.append(ret_dict)
    
    return processed_batch


def ray_collate_batch(batch_list):
    """
    Ray Data compatible collate function for MTR data
    """
    import torch
    import numpy as np
    from mtr.utils import common_utils
    
    if not batch_list or len(batch_list) == 0:
        return {}
    
    # Flatten batch_list if it's nested
    flat_batch = []
    for item in batch_list:
        if isinstance(item, list):
            flat_batch.extend(item)
        else:
            flat_batch.append(item)
    
    if not flat_batch:
        return {}
    
    batch_size = len(flat_batch)
    key_to_list = {}
    
    # Collect all keys from all items
    for key in flat_batch[0].keys():
        key_to_list[key] = [flat_batch[bs_idx][key] for bs_idx in range(batch_size)]

    input_dict = {}
    for key, val_list in key_to_list.items():
        try:
            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
                val_list = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0) if len(val_list) > 0 else torch.empty(0)
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            # Skip problematic keys
            continue
    
    # Add batch size information
    input_dict['batch_size'] = batch_size
    return input_dict


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
    
    # Get Ray dataset from trainer
    train_dataset = ray.train.get_dataset_shard("train")
    
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
    
    # Estimate iterations per epoch
    try:
        dataset_size = train_dataset.count()
        estimated_iters_per_epoch = dataset_size // worker_batch_size
    except:
        estimated_iters_per_epoch = 1000  # Default estimate
    
    # Build scheduler
    scheduler = build_scheduler(
        optimizer, cfg.OPTIMIZATION, total_epochs=args.epochs,
        total_iters_each_epoch=estimated_iters_per_epoch, last_epoch=start_epoch-1
    )
    
    # Training loop
    accumulated_iter = start_iter
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_idx = 0
        
        # Create iterator for this epoch
        train_data_iterator = train_dataset.iter_torch_batches(
            batch_size=worker_batch_size,
            drop_last=True
        )
        
        # Iterate through batches
        for batch in train_data_iterator:
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
            
            # Move batch to device if needed
            device = ray.train.torch.get_device()
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(device)
            
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
            batch_idx += 1
            
            # Log progress
            if rank == 0 and (batch_idx % args.logger_iter_interval == 0):
                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                disp_str += f', lr={cur_lr:.6f}'
                batch_size = batch.get('batch_size', worker_batch_size)
                if logger:
                    logger.info(f'epoch: {epoch}/{args.epochs}, iter: {batch_idx}, '
                              f'batch_size: {batch_size}, accumulated_iter: {accumulated_iter}, {disp_str}')
            
            # Break after reasonable number of batches per epoch
            if batch_idx >= estimated_iters_per_epoch:
                break
        
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
    
    # Update cfg with S3 data paths
    if args.s3_data_root:
        cfg.DATA_CONFIG.DATA_ROOT = args.s3_data_root
    
    # Create global batch size
    global_batch_size = args.batch_size if args.batch_size else cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU * args.num_workers
    worker_batch_size = global_batch_size // args.num_workers
    
    print("**********************Creating Ray Dataset**********************")
    
    # Create Ray dataset from scenario files
    data_root = cfg.DATA_CONFIG.DATA_ROOT
    info_file = cfg.DATA_CONFIG.INFO_FILE['train']
    
    ray_dataset = create_ray_dataset_from_scenario_files(
        data_root=data_root,
        info_file=info_file,
        dataset_cfg=cfg.DATA_CONFIG,
        training=True
    )
    
    # Process scenarios and create batches
    ray_dataset = ray_dataset.map_batches(
        load_and_process_scenario,
        batch_size=1,  # Process one scenario at a time
        batch_format="default"
    )
    
    # Create final batched dataset
    ray_dataset = ray_dataset.map_batches(
        ray_collate_batch,
        batch_size=worker_batch_size,
        batch_format="default"
    )
    
    print(f"Global batch size: {global_batch_size}, Worker batch size: {worker_batch_size}")
    
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
    
    # Create and run trainer with Ray dataset
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": ray_dataset}  # Pass the Ray dataset
    )
    
    print("**********************Starting Ray Training**********************")
    result = trainer.fit()
    
    print("**********************Training Completed**********************")
    print(f"Best checkpoint: {result.checkpoint}")
    print(f"Final metrics: {result.metrics}")
    print(f"Training logs path: {result.path}")


if __name__ == '__main__':
    main()
