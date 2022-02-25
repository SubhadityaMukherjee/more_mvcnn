#!/bin/bash
#run1
EPOCHS=20
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=128 --epochs=$EPOCHS --architecture="mobilenet" --lr=1e-4 
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=128 --epochs=$EPOCHS --architecture="mobilenetv2" --lr=1e-4 
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=128 --epochs=$EPOCHS --architecture="vgg" --lr=1e-4
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=128 --epochs=$EPOCHS --architecture="efficientnet" --lr=1e-4
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=128 --epochs=$EPOCHS --architecture="xception" --lr=1e-4



