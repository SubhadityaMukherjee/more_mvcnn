#!/bin/bash
#SBATCH --job-name="mvc"
#SBATCH --time=10:00:00
#SBATCH --partition=regular
#SBATCH --nodes=3
#SBATCH --mem=5G

module purge
module load Python
#python3 /home/s4747925/more_mvcnn/prevoxelization.py --modelnet10 /data/s4747925/data/ModelNet40
singularity exec -B /data/s4747925/data/ /data/s4747925/data/open3d.sif "python3" "/home/s4747925/more_mvcnn/prevoxelization.py" "--modelnet10" "/data/s4747925/data/ModelNet40"

###SBATCH --gres=gpu:v100:1
