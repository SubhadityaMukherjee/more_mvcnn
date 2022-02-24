#!/bin/bash

# python3 generate_entropy_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40 --out data/entropy_dataset/
#echo "DONE_ENTROPY"
# python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40 --set train --out data/
# python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40 --set test --out data/
python3 generate_entropy_dataset_p.py --modelnet10 /media/hdd/Datasets/ModelNet40 --set train --out data/
python3 generate_entropy_dataset_p.py --modelnet10 /media/hdd/Datasets/ModelNet40 --set test --out data/

#echo "DONE_VIEW"

# python3 prevoxelization.py --modelnet10 /media/hdd/Datasets/ModelNet40
# echo "DONE_VOXEL"

# python3 entropy_model.py --voxel_data voxel_data --entropy_dataset data/entropy_dataset/entropy_dataset.csv --epochs 1 --batch_size 8 --out data
