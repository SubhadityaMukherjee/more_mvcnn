#!/bin/bash
#SBATCH --job-name="mvc"
#SBATCH --time=10:00:00
#SBATCH --partition=regular
#SBATCH --nodes=3
#SBATCH --mem=5G

module purge
module load Python
singularity exec -B /data/username/data/ /data/username/data/open3d.sif "python3" "/home/username/more_mvcnn/prevoxelization.py" "--modelnet10" "/data/username/data/ModelNet40"

###SBATCH --gres=gpu:v100:1
