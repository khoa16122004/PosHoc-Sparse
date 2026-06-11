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
echo "Job $SLURM_JOB_ID starts on GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

cd /datastore/elo/khoatn/PosHoc-Sparse || exit 1

IMAGE_PATH="imgs/dog_cat.png"
TEXT_PROMPT="dog"
MODEL_NAME="${3:-openai/clip-vit-base-patch16}"
METHOD="${4:-attn_grad}"
OUTPUT_DIR="${5:-saliency_result/single_clip}"
OVERLAY_ALPHA="${6:-0.45}"

if [ -z "$IMAGE_PATH" ] || [ -z "$TEXT_PROMPT" ]; then
    echo "Usage: sbatch script_sh/export_explain_clip_text.sh <image_path> <text_prompt> [model_name] [method] [output_dir] [overlay_alpha]"
    exit 1
fi

echo "Running CLIP single-image explanation"
echo "image=$IMAGE_PATH"
echo "text=$TEXT_PROMPT"
echo "model=$MODEL_NAME"
echo "method=$METHOD"

python script/explain_clip_by_text.py \
    --image-path "$IMAGE_PATH" \
    --text "$TEXT_PROMPT" \
    --model-name "$MODEL_NAME" \
    --method "$METHOD" \
    --output-dir "$OUTPUT_DIR" \
    --overlay-alpha "$OVERLAY_ALPHA"
