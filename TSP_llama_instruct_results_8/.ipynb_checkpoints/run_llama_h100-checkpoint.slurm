#!/bin/bash
#SBATCH --job-name=Python_Script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/xhu/COSC426/llama_h100_out.log
#SBATCH --error=/home/xhu/COSC426/llama_h100_error.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xhu@colgate.edu
#SBATCH --export=ALL

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Node list: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd $SLURM_SUBMIT_DIR

source /local/JupyterHub/bin/activate && conda activate nlp

echo "=== GPU STATUS CHECK ==="
nvidia-smi || echo "nvidia-smi failed — check node GPU availability"

python - <<EOF
import torch
if torch.cuda.is_available():
    print("Number of GPUs detected:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"  Reserved memory:  {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
else:
    print("No GPU available in PyTorch")
EOF

python /home/xhu/NLPScholar_final/main.py /home/xhu/COSC426/llama_inst_suboptimal.yaml
