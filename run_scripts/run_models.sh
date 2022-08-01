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
# BTS=512
# EPOCHS=10
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_10" --train_sample_ratio 10 --test_sample_ratio 100
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_1_20" --train_sample_ratio 10 --test_sample_ratio 200

# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "m40_mobilenet_1_10" --train_sample_ratio 10 --test_sample_ratio 100 --modeln "modelnet40"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "m40_mobilenet_1_20" --train_sample_ratio 10 --test_sample_ratio 200 --modeln "modelnet40"

# BTS=128
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_1_10" --train_sample_ratio 10 --test_sample_ratio 100
# python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-test-m10/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_1_20" --train_sample_ratio 10 --test_sample_ratio 200

# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m40_vgg_1_10" --train_sample_ratio 10 --test_sample_ratio 100 --modeln "modelnet40"
# python3 single_view_cnn.py --train_data=data/view-dataset-train/image/ --test_data=data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m40_vgg_1_20" --train_sample_ratio 10 --test_sample_ratio 200 --modeln "modelnet40"


# Deformed Test set

#voxsize 0.2 0.4 0.6 0.8
# EPOCHS=1
# declare -a voxar=(0.02 0.04 0.06 0.08 0.10)
# for VOXSIZE in "${voxar[@]}"
# do
#     rm -rf data/view-dataset-deformed/image/
#     export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset_deformed.py --modelnet10 /media/hdd/Datasets/ModelNet10 --set test --out data/ --mname "modelnet10" --voxsize $VOXSIZE
#     BTS=512
#     python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-deformed/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_vox_$VOXSIZE" --train_sample_ratio 100 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/mobilenetv2-27-Feb-233844-m10_mobilenetv2_1_1/classification_model.h5"
#     BTS=128
#     python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-deformed/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_vox_$VOXSIZE" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/vgg-28-Feb-033836-m10_vgg_1_10/classification_model.h5"

#     rm -rf data/view-dataset-deformed/
# done


# SIGMA=0.2

# declare -a sigarr=(0.002 0.004 0.006 0.008 0.010)
# declare -a sigarr=(0.002)

# for SIGMA in "${sigarr[@]}"
# do

#     rm -rf data/view-dataset-deformed/image/
#     export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 generate_view_dataset_deformed.py --modelnet10 /media/hdd/Datasets/ModelNet10 --set test --out data/ --mname "modelnet10" --sigma $SIGMA
#     BTS=512
#     python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-deformed/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "m10_mobilenetv2_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/mobilenetv2-27-Feb-233844-m10_mobilenetv2_1_1/classification_model.h5" --sigma $SIGMA
#     BTS=128
#     python3 single_view_cnn.py --train_data=data/view-dataset-train-m10/image/ --test_data=data/view-dataset-deformed/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "m10_vgg_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/logs/vgg-28-Feb-033836-m10_vgg_1_10/classification_model.h5" --sigma $SIGMA
#     rm -rf data/view-dataset-deformed/image/
# done

# "/media/hdd/github/more_mvcnn/logs/mobilenetv2-27-Feb-233844-m10_mobilenetv2_1_1/classification_model.h5" 
# "/media/hdd/github/more_mvcnn/logs/vgg-28-Feb-033836-m10_vgg_1_10/classification_model.h5" 

# "/media/hdd/github/more_mvcnn/logs/vgg-28-Feb-035400-m40_vgg_1_10/classification_model.h5" 
# "/media/hdd/github/more_mvcnn/logs/mobilenet-28-Feb-030442-m40_mobilenet_1_20/classification_model.h5" 

# Check for differnt sigma values
# declare -a sigarr=(0.02 0.04 0.06 0.08 0.1)
# BTS=128
# EPOCHS=1

# for SIGMA in "${sigarr[@]}"
# do
#     # python3 single_view_cnn.py --train_data=old_data/view-dataset-train-m10/image/ --test_data=view-dataset-deformed/modelnet10/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "new_m10_vgg_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/vgg-28-Feb-033836-m10_vgg_1_10/classification_model.h5" --sigma $SIGMA --getpred
#     # python3 single_view_cnn.py --train_data=old_data/view-dataset-train-m10/image/ --test_data=view-dataset-deformed/modelnet10/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "new_m10_mobilenetv2_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/mobilenetv2-27-Feb-233844-m10_mobilenetv2_1_1/classification_model.h5" --sigma $SIGMA --getpred

#     # python3 single_view_cnn.py --train_data=old_data/view-dataset-train/image/ --test_data=view-dataset-deformed/modelnet40/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "new_m40_vgg_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/vgg-28-Feb-035400-m40_vgg_1_10/classification_model.h5" --sigma $SIGMA --getpred --modeln "modelnet40"
#     python3 single_view_cnn.py --train_data=old_data/view-dataset-train/image/ --test_data=view-dataset-deformed/modelnet40/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenet" --name "new_m40_mobilenetv2_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/mobilenet-26-Feb-192611/classification_model.h5" --sigma $SIGMA --getpred --modeln "modelnet40"

# done

BTS=1
EPOCHS=1
SIGMA=0.0

# python3 single_view_cnn.py --train_data=old_data/view-dataset-train-m10/image/ --test_data=view-dataset-deformed/modelnet10/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="mobilenetv2" --name "new_m10_mobilenetv2_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_scripts/old_data/classifier_model_mnet10.h5" --sigma $SIGMA --getpred

# python3 single_view_cnn.py --train_data=old_data/view-dataset-train-m10/image/ --test_data=view-dataset-deformed/modelnet10/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "new_m10_vgg_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/vgg-28-Feb-033836-m10_vgg_1_10/classification_model.h5" --sigma $SIGMA --getpred

# python3 single_view_cnn.py --train_data=old_data/view-dataset-train/image/ --test_data=view-dataset-deformed/modelnet40/image/$SIGMA --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "new_m40_vgg_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/vgg-28-Feb-035400-m40_vgg_1_10/classification_model.h5" --sigma $SIGMA --getpred --modeln "modelnet40"
# python3 single_view_cnn.py --train_data=old_data/view-dataset-train/image/ --test_data=old_data/view-dataset-test/image --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "new_m40_vgg_sig" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/vgg-28-Feb-035400-m40_vgg_1_10/classification_model.h5" --getpred --modeln "modelnet40"
# python3 single_view_cnn.py --train_data=old_data/view-dataset-train/image/ --test_data=old_data/view-dataset-test/image/ --batch_size=$BTS --epochs=$EPOCHS --architecture="vgg" --name "new_m40_vgg_sig_$SIGMA" --train_sample_ratio 10 --test_sample_ratio 10 --load_model "/media/hdd/github/more_mvcnn/old_logs/vgg-28-Feb-035400-m40_vgg_1_10/classification_model.h5" --getpred --modeln "modelnet40"


# export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 multi_view_demo.py --data "/media/hdd/Datasets/ModelNet10/bathtub/test/bathtub_0107.off" --entropy_model "old_data/entropy_model_mnet10.h5" --classifier_model "old_data/classifier_model_mnet10.h5"
export MESA_LOADER_DRIVER_OVERRIDE=i965;python3 evaluation.py --modelnet10 "/media/hdd/Datasets/ModelNet10" --entropy_model "old_data/entropy_model_mnet10.h5" --classifier_model "old_data/classifier_model_mnet10.h5" --name "mnet10_entropy_model" --view_dataset "old_data/view-dataset-deformed/modelnet10/image/0.0/"