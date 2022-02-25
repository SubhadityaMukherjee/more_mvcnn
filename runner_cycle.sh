#!/bin/bash
#SBATCH --job-name="mvc"
#SBATCH --time=00:5:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4G
#SBATCH --nodes=3

#module purge
#module load Python 
python3 /home/s4747925/more_mvcnn/prevoxelization.py --modelnet10 /data/s4747925/data/ModelNet40

singularity exec -B /data/s4747925/data/ /data/s4747925/data/open3d.sif "python3" 
