#!/bin/bash
#SBATCH --job-name=SPAS
#SBATCH --output=revise/mps_%j.out
#SBATCH --error=revise/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2
#SBATCH --mem=4G
#SBATCH --time=72:00:00

REQUIRED_VRAM=20000

module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack
echo "ENV:" $CONDA_DEFAULT_ENV
echo "PREFIX:" $CONDA_PREFIX
which python
python -c "import sys; print(sys.executable)"

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

cd /datastore/elo/khoatn/PosHoc-Sparse || exit 1

VAL_DIR=/datastore/elo/quanphm/dataset/ImageNet1K/val/
OUTPUT_DIR=saliency_results
TARGET_SOURCE=original_class
OVERLAY_ALPHA=0.45

MODELS=(
    "resnet18"
    "densenet121"
    "vgg16"
)

METHODS=("Grad" "Grad_Input" "Int_Grad")

for model_name in "${MODELS[@]}"; do
    input_json="evaluate_results/torchvision/$model_name/selected_1000.json"
    if [ ! -f "$input_json" ]; then
        echo "Skip missing input: $input_json"
        continue
    fi

    for method in "${METHODS[@]}"; do
        echo "Running torchvision | model=$model_name | method=$method"
        python script/export_single_explain.py \
            --input-json "$input_json" \
            --val-dir "$VAL_DIR" \
            --model-name "$model_name" \
            --method "$method" \
            --type torchvision \
            --target-source "$TARGET_SOURCE" \
            --overlay-alpha "$OVERLAY_ALPHA" \
            --output-dir "$OUTPUT_DIR"
    done
done