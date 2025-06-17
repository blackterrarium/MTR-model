#!/bin/bash

# Ray Training Launcher Script for MTR Model
# This script provides convenient commands for different Ray training scenarios

set -e

# Default configuration
DEFAULT_CONFIG="tools/cfgs/waymo/mtr_ray_training.yaml"
DEFAULT_WORKERS=4
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCHS=50
DEFAULT_EXPERIMENT="mtr_ray_experiment"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if Ray is running
check_ray_status() {
    if command -v ray &> /dev/null; then
        if ray status &> /dev/null; then
            print_success "Ray cluster is running"
            ray status
            return 0
        else
            print_warning "Ray is installed but no cluster is running"
            return 1
        fi
    else
        print_error "Ray is not installed"
        return 1
    fi
}

# Function to start local Ray cluster
start_local_ray() {
    print_info "Starting local Ray cluster..."
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --object-store-memory=10000000000
    print_success "Ray cluster started. Dashboard available at http://localhost:8265"
}

# Function to stop Ray cluster
stop_ray() {
    print_info "Stopping Ray cluster..."
    ray stop
    print_success "Ray cluster stopped"
}

# Function to validate environment
validate_environment() {
    print_info "Validating environment..."
    
    # Check Python packages
    python -c "import ray; print(f'Ray version: {ray.__version__}')" || {
        print_error "Ray is not installed. Run: pip install -U 'ray[train]'"
        exit 1
    }
    
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
        print_error "PyTorch is not installed"
        exit 1
    }
    
    # Check CUDA availability
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_success "CUDA is available"
        python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
    else
        print_warning "CUDA is not available, training will use CPU"
    fi
    
    # Check S3 credentials
    if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]]; then
        print_success "S3 credentials are configured"
    else
        print_warning "S3 credentials not found in environment"
    fi
    
    print_success "Environment validation completed"
}

# Function to run training
run_training() {
    local config_file="${1:-$DEFAULT_CONFIG}"
    local num_workers="${2:-$DEFAULT_WORKERS}"
    local batch_size="${3:-$DEFAULT_BATCH_SIZE}"
    local epochs="${4:-$DEFAULT_EPOCHS}"
    local experiment_name="${5:-$DEFAULT_EXPERIMENT}"
    
    print_info "Starting Ray training with the following configuration:"
    echo "  Config file: $config_file"
    echo "  Workers: $num_workers"
    echo "  Batch size: $batch_size"
    echo "  Epochs: $epochs"
    echo "  Experiment: $experiment_name"
    
    # Build the command
    local cmd="python tools/train_ray.py"
    cmd="$cmd --cfg_file $config_file"
    cmd="$cmd --num_workers $num_workers"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --epochs $epochs"
    cmd="$cmd --experiment_name $experiment_name"
    cmd="$cmd --use_gpu"
    
    # Add S3 configuration if available
    if [[ -n "$MINIO_ENDPOINT" ]]; then
        cmd="$cmd --minio_endpoint $MINIO_ENDPOINT"
    fi
    if [[ -n "$MINIO_ACCESS_KEY" ]]; then
        cmd="$cmd --minio_access_key $MINIO_ACCESS_KEY"
    fi
    if [[ -n "$MINIO_SECRET_KEY" ]]; then
        cmd="$cmd --minio_secret_key $MINIO_SECRET_KEY"
    fi
    if [[ -n "$S3_DATA_ROOT" ]]; then
        cmd="$cmd --s3_data_root $S3_DATA_ROOT"
    fi
    if [[ -n "$STORAGE_PATH" ]]; then
        cmd="$cmd --storage_path $STORAGE_PATH"
    fi
    
    print_info "Executing command: $cmd"
    eval $cmd
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start-ray              Start a local Ray cluster"
    echo "  stop-ray               Stop the Ray cluster"
    echo "  status                 Check Ray cluster status"
    echo "  validate               Validate the environment"
    echo "  train                  Run training (default command)"
    echo "  quick-train            Run training with minimal configuration"
    echo "  distributed-train      Run distributed training with multiple workers"
    echo "  help                   Show this help message"
    echo ""
    echo "Training Options:"
    echo "  --config CONFIG_FILE   Configuration file (default: $DEFAULT_CONFIG)"
    echo "  --workers NUM          Number of Ray workers (default: $DEFAULT_WORKERS)"
    echo "  --batch-size SIZE      Global batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --epochs NUM           Number of epochs (default: $DEFAULT_EPOCHS)"
    echo "  --experiment NAME      Experiment name (default: $DEFAULT_EXPERIMENT)"
    echo ""
    echo "Environment Variables:"
    echo "  MINIO_ENDPOINT         MinIO server endpoint (e.g., http://localhost:9000)"
    echo "  MINIO_ACCESS_KEY       MinIO access key"
    echo "  MINIO_SECRET_KEY       MinIO secret key"
    echo "  S3_DATA_ROOT          S3 path for training data (e.g., s3://mtr-data/waymo)"
    echo "  STORAGE_PATH          Ray storage path (e.g., s3://mtr-training)"
    echo ""
    echo "Examples:"
    echo "  $0 validate                    # Check environment"
    echo "  $0 start-ray                   # Start local Ray cluster"
    echo "  $0 quick-train                 # Quick training with 2 workers"
    echo "  $0 train --workers 8 --epochs 100  # Custom training"
    echo ""
    echo "For distributed training across multiple machines:"
    echo "  # On head node:"
    echo "  ray start --head --dashboard-host=0.0.0.0"
    echo "  # On worker nodes:"
    echo "  ray start --address='head-node-ip:6379'"
    echo "  # Then run training:"
    echo "  $0 distributed-train"
}

# Parse command line arguments
COMMAND="${1:-train}"
shift || true

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    start-ray)
        start_local_ray
        ;;
    stop-ray)
        stop_ray
        ;;
    status)
        check_ray_status
        ;;
    validate)
        validate_environment
        ;;
    train)
        validate_environment
        if ! check_ray_status; then
            print_info "Ray cluster not running. Starting local cluster..."
            start_local_ray
        fi
        run_training "${CONFIG_FILE}" "${NUM_WORKERS}" "${BATCH_SIZE}" "${EPOCHS}" "${EXPERIMENT_NAME}"
        ;;
    quick-train)
        validate_environment
        if ! check_ray_status; then
            start_local_ray
        fi
        run_training "${CONFIG_FILE:-$DEFAULT_CONFIG}" 2 16 10 "quick_experiment"
        ;;
    distributed-train)
        validate_environment
        if ! check_ray_status; then
            print_error "Ray cluster is not running. Start Ray cluster first."
            exit 1
        fi
        run_training "${CONFIG_FILE}" "${NUM_WORKERS:-8}" "${BATCH_SIZE:-64}" "${EPOCHS}" "${EXPERIMENT_NAME:-distributed_experiment}"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
