#!/bin/bash
# export mesa_loader_driver_override=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/datasets/modelnet40 --set train --out data/
# export mesa_loader_driver_override=i965;python3 generate_view_dataset.py --modelnet10 /media/hdd/datasets/modelnet40 --set test --out data/
#run1
# EPOCHS=40
# BTS=128
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="efficientnet"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg"

# Top-k modelnet10
# mobilenet
# BTS=512
# EPOCHS=6
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_1" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/mobilenetv2-27-Feb-131551/classification_model.h5"
# EPOCHS=1
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_3" --train_sample_ratio 10 --test_sample_ratio 30 --load_model "/media/hdd/github/more_mvcnn/logs/mobilenetv2-27-Feb-131551/classification_model.h5"

# vgg

# BTS=128
# EPOCHS=6
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_1_1" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/vgg-27-Feb-124545/classification_model.h5"
# EPOCHS=1
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_1_3" --train_sample_ratio 10 --test_sample_ratio 30 --load_model "/media/hdd/github/more_mvcnn/logs/vgg-27-Feb-124545/classification_model.h5"

#Top-k modelnet40
#mobilenet

# BTS=512
# EPOCHS=3
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "m40_mobilenet_1_1" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/mobilenet-26-Feb-192611/classification_model.h5" --modeln "modelnet40"
# EPOCHS=1
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "m40_mobilenet_1_3" --train_sample_ratio 10 --test_sample_ratio 30 --load_model "/media/hdd/github/more_mvcnn/logs/mobilenet-26-Feb-192611/classification_model.h5" --modeln "modelnet40"

# # vgg

# BTS=128
# EPOCHS=3
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m40_vgg_1_1" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/vgg-27-Feb-104726/classification_model.h5"  --modeln "modelnet40"
# EPOCHS=1
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m40_vgg_1_3" --train_sample_ratio 10 --test_sample_ratio 30 --load_model "/media/hdd/github/more_mvcnn/logs/vgg-27-Feb-104726/classification_model.h5" --modeln "modelnet40"


# 10, 20 : ModelNet10
BTS=512
EPOCHS=30
python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_10" --train_sample_ratio 10 --test_sample_ratio 100
python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_20" --train_sample_ratio 10 --test_sample_ratio 200

python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "m40_mobilenet_1_10" --train_sample_ratio 10 --test_sample_ratio 100 --modeln "modelnet40"
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "m40_mobilenet_1_20" --train_sample_ratio 10 --test_sample_ratio 200 --modeln "modelnet40"

BTS=128
python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_1_10" --train_sample_ratio 10 --test_sample_ratio 100
python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_1_20" --train_sample_ratio 10 --test_sample_ratio 200

python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m40_vgg_1_10" --train_sample_ratio 10 --test_sample_ratio 100 --modeln "modelnet40"
python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m40_vgg_1_20" --train_sample_ratio 10 --test_sample_ratio 200 --modeln "modelnet40"

# BTS=512
# EPOCHS=230
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_20" --train_sample_ratio 10 --test_sample_ratio 200
