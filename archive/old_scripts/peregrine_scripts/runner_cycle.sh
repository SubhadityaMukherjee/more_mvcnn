#!/bin/bash
#SBATCH --job-name="mvc"
#SBATCH --time=00:8:00
#SBATCH --partition=regular
#SBATCH --mem=10G

module load R/4.0.0-foss-2020a
module list
Rscript -e 'install.packages("abc")'
Rscript -e 'install.packages("abc")'
Rscript -e 'install.packages("abc")'
Rscript scripttorun.r
