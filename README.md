# CAGNet
The pytorch implementation for CAGNet in paper "Cross-Attention Guided Group Aggregation Network for Cropland Change Detection".
# Requirements
* Python 3.6
* Pytorch 1.7.0

# Datasets Preparation
The path list in the datasest folder is as follows:

|—train

* ||—A

* ||—B

* ||—OUT

|—val

* ||—A

* ||—B

* ||—OUT

|—test

* ||—A

* ||—B

* ||—OUT


where A contains pre-temporal images, B contains post-temporal images, and OUT contains ground truth images.
# Train
* python train.py --dataset-dir dataset-path
# Test
* python eval.py --ckp-paths weight-path --dataset-dir dataset-path
# Visualization
* python visualization visualization.py --ckp-paths weight-path --dataset-dir dataset-path (Note that batch-size must be 1 when using visualization.py)
* Besides, you can adjust the parameter of full_to_color to change the color

* C. Xu et al., "Cross-Attention Guided Group Aggregation Network for Cropland Change Detection," in IEEE Sensors Journal, vol. 23, no. 12, pp. 13680-13691, 15 June15, 2023, doi: 10.1109/JSEN.2023.3271391.
