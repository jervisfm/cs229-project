# This is a script to run some experiments for transfer learning with different transfer model.

# Note this assumes that you've pre-downloaded the data for the different classes
# already.

# Run first experiment on 3 classes.

set -x
MAX_ITER=100
echo "Running transfer learning experiments on 3 classes!"

# echo "Transfer learning with MobileNet"
# python3 transfer_learning.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_3/ --transfer_model=MobileNet --experiment_name="transfer_learning_with_MobileNet_3_classes" "$@"

echo "Transfer learning with InceptionV3"
python3 transfer_learning.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_3/ --transfer_model=InceptionV3 --experiment_name="transfer_learning_with_InceptionV3_3_classes" "$@"

echo "Transfer learning with ResNet50"
python3 transfer_learning.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_3/ --transfer_model=ResNet50 --experiment_name="transfer_learning_with_ResNet50_3_classes" "$@"

echo "Transfer learning with VGG19"
python3 transfer_learning.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_3/ --transfer_model=VGG19 --experiment_name="transfer_learning_with_VGG19_3_classes" "$@"

# All done
echo "All done!"
