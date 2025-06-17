#!/bin/bash

# MinIO Setup Script for MTR Data
# This script helps set up MinIO and upload data for Ray training

set -e

# Configuration
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minio_access_key}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minio_secret_key}"
MC_ALIAS="myminio"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if MinIO client is available
check_mc() {
    if ! command -v mc &> /dev/null; then
        print_info "MinIO client not found. Installing..."
        
        # Download MinIO client
        case "$(uname -s)" in
            Linux*)
                wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O mc
                ;;
            Darwin*)
                wget -q https://dl.min.io/client/mc/release/darwin-amd64/mc -O mc
                ;;
            *)
                print_error "Unsupported operating system"
                exit 1
                ;;
        esac
        
        chmod +x mc
        sudo mv mc /usr/local/bin/
        print_success "MinIO client installed"
    fi
}

# Configure MinIO client
configure_mc() {
    print_info "Configuring MinIO client..."
    mc alias set $MC_ALIAS $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY
    
    # Test connection
    if mc ls $MC_ALIAS > /dev/null 2>&1; then
        print_success "Successfully connected to MinIO at $MINIO_ENDPOINT"
    else
        print_error "Failed to connect to MinIO. Check your configuration."
        exit 1
    fi
}

# Create buckets
create_buckets() {
    print_info "Creating buckets..."
    
    # Create data bucket
    if mc mb $MC_ALIAS/mtr-data --ignore-existing; then
        print_success "Created bucket: mtr-data"
    else
        print_warning "Bucket mtr-data already exists or failed to create"
    fi
    
    # Create training bucket
    if mc mb $MC_ALIAS/mtr-training --ignore-existing; then
        print_success "Created bucket: mtr-training"
    else
        print_warning "Bucket mtr-training already exists or failed to create"
    fi
    
    # List buckets
    print_info "Available buckets:"
    mc ls $MC_ALIAS
}

# Upload data
upload_data() {
    local data_dir="${1:-./data}"
    
    if [[ ! -d "$data_dir" ]]; then
        print_error "Data directory not found: $data_dir"
        return 1
    fi
    
    print_info "Uploading data from $data_dir to MinIO..."
    
    # Upload Waymo data
    if [[ -d "$data_dir/waymo" ]]; then
        print_info "Uploading Waymo dataset..."
        mc cp --recursive "$data_dir/waymo/" $MC_ALIAS/mtr-data/waymo/
        print_success "Waymo data uploaded"
    else
        print_warning "Waymo data directory not found in $data_dir"
    fi
    
    # Show uploaded data structure
    print_info "Data structure in MinIO:"
    mc tree $MC_ALIAS/mtr-data
}

# Download data
download_data() {
    local output_dir="${1:-./data_from_s3}"
    
    print_info "Downloading data from MinIO to $output_dir..."
    mkdir -p "$output_dir"
    
    mc cp --recursive $MC_ALIAS/mtr-data/ "$output_dir/"
    print_success "Data downloaded to $output_dir"
}

# Set bucket policy for public access (if needed)
set_public_policy() {
    local bucket="$1"
    
    print_info "Setting public read policy for bucket: $bucket"
    
    cat > /tmp/policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "*"},
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::$bucket/*"]
    }
  ]
}
EOF
    
    mc policy set-json /tmp/policy.json $MC_ALIAS/$bucket
    rm /tmp/policy.json
    print_success "Public read policy set for $bucket"
}

# Show bucket information
show_info() {
    print_info "MinIO Configuration:"
    echo "  Endpoint: $MINIO_ENDPOINT"
    echo "  Access Key: $MINIO_ACCESS_KEY"
    echo "  Secret Key: ${MINIO_SECRET_KEY:0:4}***"
    echo ""
    
    print_info "Available buckets:"
    mc ls $MC_ALIAS
    echo ""
    
    print_info "Data bucket contents:"
    mc ls --recursive $MC_ALIAS/mtr-data | head -20
    if [[ $(mc ls --recursive $MC_ALIAS/mtr-data | wc -l) -gt 20 ]]; then
        echo "... (truncated, showing first 20 items)"
    fi
}

# Cleanup function
cleanup() {
    print_warning "Cleaning up buckets (this will delete all data)..."
    read -p "Are you sure you want to delete all data? (yes/no): " confirm
    
    if [[ $confirm == "yes" ]]; then
        mc rm --recursive --force $MC_ALIAS/mtr-data
        mc rm --recursive --force $MC_ALIAS/mtr-training
        mc rb $MC_ALIAS/mtr-data
        mc rb $MC_ALIAS/mtr-training
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Show usage
show_usage() {
    echo "MinIO Setup Script for MTR Training"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                  Complete setup (configure + create buckets)"
    echo "  configure              Configure MinIO client"
    echo "  create-buckets         Create required buckets"
    echo "  upload [DATA_DIR]      Upload data to MinIO (default: ./data)"
    echo "  download [OUTPUT_DIR]  Download data from MinIO (default: ./data_from_s3)"
    echo "  info                   Show MinIO configuration and bucket contents"
    echo "  cleanup                Delete all buckets and data"
    echo "  public [BUCKET]        Set bucket to public read access"
    echo "  help                   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MINIO_ENDPOINT         MinIO server endpoint (default: http://localhost:9000)"
    echo "  MINIO_ACCESS_KEY       MinIO access key (default: minio_access_key)"
    echo "  MINIO_SECRET_KEY       MinIO secret key (default: minio_secret_key)"
    echo ""
    echo "Examples:"
    echo "  $0 setup                        # Complete setup"
    echo "  $0 upload ./my_data             # Upload data from custom directory"
    echo "  $0 public mtr-data              # Make data bucket publicly readable"
    echo "  MINIO_ENDPOINT=http://remote:9000 $0 configure  # Connect to remote MinIO"
}

# Main command handling
case "${1:-setup}" in
    setup)
        check_mc
        configure_mc
        create_buckets
        print_success "MinIO setup completed!"
        print_info "You can now upload data with: $0 upload [data_directory]"
        ;;
    configure)
        check_mc
        configure_mc
        ;;
    create-buckets)
        check_mc
        configure_mc
        create_buckets
        ;;
    upload)
        check_mc
        configure_mc
        upload_data "$2"
        ;;
    download)
        check_mc
        configure_mc
        download_data "$2"
        ;;
    info)
        check_mc
        configure_mc
        show_info
        ;;
    public)
        check_mc
        configure_mc
        set_public_policy "${2:-mtr-data}"
        ;;
    cleanup)
        check_mc
        configure_mc
        cleanup
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
