#!/bin/bash

python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=64 --epochs=30 --architecture="vgg" --lr=1e-4
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=64 --epochs=30 --architecture="efficientnet" --lr=1e-4
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=64 --epochs=30 --architecture="mobilenet" --lr=1e-4
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=64 --epochs=30 --architecture="mobilenetv2" --lr=1e-4