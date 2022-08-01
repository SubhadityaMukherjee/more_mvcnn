# Generate view dataset for Modelnet10
# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet10/ --set train --out data/modelnet10/
# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet10/ --set test --out data/modelnet10/

# Generate view dataset for Modelnet40
# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40/ --set train --out data/modelnet40/
# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40/ --set test --out data/modelnet40/

# Run entropy model
rm -rf ./logs/entropy_model/
export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 entropy_model.py --voxel_data old_scripts/.voxel_data --entropy_dataset old_scripts/old_data/entropy-dataset-train/entropy_dataset.csv 