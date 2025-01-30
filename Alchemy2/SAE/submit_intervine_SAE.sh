#!/bin/bash
#SBATCH --job-name=llama_prediction            # Job name
#SBATCH --output=llama_prediction_%j.out       # Standard output log (%j will be replaced by job ID)
#SBATCH --error=llama_prediction_%j.err        # Standard error log (%j will be replaced by job ID)
#SBATCH --account=gts-rwilson337-prepaid       # Account name
#SBATCH --gres=gpu:A100:1                      # Request a single A100 GPU                              
#SBATCH -C A100-80GB                           # Request a single A100 GPU with 80GB memory
#SBATCH --mem-per-gpu=64G                      # Memory allocated per GPU
#SBATCH --cpus-per-task=8                      # Number of CPU cores per task
#SBATCH --time=24:00:00                        # Time limit hrs:min:sec
#SBATCH --tmp=120G                              # Temporary disk storage
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail notifications for job events
#SBATCH --mail-user=hanboxie1997@gatech.edu

# Load Anaconda module (adjust the module name if needed)
module load anaconda3/2023.03
source $(conda info --base)/etc/profile.d/conda.sh
conda activate think_aloud  # Activate your environment

# Navigate to the directory where the code is located
cd /storage/coda1/p-rwilson337/0/hxie88/LLMs_game/Alchemy2/SAE
export HF_HOME=/storage/coda1/p-rwilson337/0/hxie88/hf_cache

srun python intervine_SAE.py --model_size 70B --variable cbu_value




