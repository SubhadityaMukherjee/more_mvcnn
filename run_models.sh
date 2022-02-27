#!/bin/bash
# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40 --set train --out data/
# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/Datasets/ModelNet40 --set test --out data/
#run1
EPOCHS=40
BTS=512
# BTS=128
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="efficientnet"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg"

# Top-k modelnet10
EPOCHS=40
python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2"

