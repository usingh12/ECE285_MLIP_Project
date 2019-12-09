# ECE 285 UCSD - Fall '19 - Final project
# TeamYOLO - Comparison of Object Detection Techniques 

## In this project we have implemented Faster-RCNN and Single-Shot Detector(SSD) for multi-object detection.
# New York City Walking Tour Videostream (Input)
[![Video stream that was used as input for detection](https://imgur.com/1hcwxrk.gif) ](https://www.youtube.com/watch?v=u68EWmtKZw0) 

# Videostream after Detection
You can find the clean files named as Faster_RCNN_Video.avi and *.avi

# Faster RCNN
A Pytorch implementation of the Faster-RCNN algorithm developed by Ross Girschick et al., based on the 2015 paper, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" pubilshed in CVPR 2015. 

# Single Shot Detector 
A PyTorch implementation of the SSD Multibox Detector for image feature extraction, based on the 2016 [Arxiv](http://arxiv.org/abs/1512.02325) paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C.
## Table of contents
* [Installation](#Installation)
* [Datasets](#Datasets)
* [Demo](#Demo)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Performance](#Performance)
* [Experiments](#Experiments)
* [Directory structure](#Directory-structure)
* [References](#References)

### Installation
To install Python dependencies and modules, use <br>
```pip install -r requirements.txt``` <br>

To get the pretrained weights ready for use, download and install according to the following instructions:
- Faster-RCNN: Download from '' and put the 'checkpoint.pth.tar' file in the folder 'checkpoint_faster_rcnn'
- SSD: 

### Datasets
[2012 version](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) of Pascal VOC dataset - well known dataset for object detection/classification/segmentation. Contains 100k images for training and validation containing bounding boxes with 20 categories of objects.

### Demo
#### Faster-RCNN
- Run **Demo_FasterRCNN.ipynb** notebook to run Faster-RCNN to plot statistics, detect images and videos on PascalVOC2012 dataset (In the notebook, set :::)
### Training
Run **train_fasterrcnn.py** script file to train the Faster-RCNN model on the PascalVOC2012 dataset.
### Evaluation
Run **eval_fasterrcnn** notebook to evaluate the Faster-RCNN model on the PASCALVOC2007 test set.

#### SSD
- Run **Demo.ipynb** notebook to run Single-Shot Detection on a random image from the PascalVOC2012 dataset.
### Training
Run **SSD_train.ipynb** notebook to train the SSD model on the PascalVOC2012 dataset.
### Evaluation
Run **SSD_Eval.ipynb** notebook to evaluate the SSD model on the PascalVOC2012 validation set.
Run **SSD_Eval_Testset.ipynb** notebook to evaluate the SSD model on the PascalVOC2007 test set. (Download the PascalVOC2007 test set using `wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar` and run `tar -xvf VOCtest_06-Nov-2007.tar` in the root directory of the repository.
### Performance <br>
On [UCSD Data Science and Machine Learning Cluster](https://datahub.ucsd.edu/hub/home):

| ------------- | Faster-RCNN(32)  | Faster-RCNN(16) | SSD |
| ------------- | ---------------- | --------------- | --- |
| Training      |  60%             | 76%             | x%   | 
| Evaluation    |  46%             | 63%             | x%   |

### Directory structure
- faster_rcnn/ - files for Faster-RCNN implementation:
    - checkpoint_fasterrcnn/ - folder that stores the checkpoint files
        - config.txt - the config file for the experiment with batch size 16
        - checkpoint.pth.tar - download this file from the above mentioned Google Drive link and place here
    - faster_rcnn - directory storing libraries for faster_rcnn implementation
        - pycache - .pyc files for Python interpreter to compile the source to
        - lib - library files for Faster-RCNN implementation
            - backbone_utils.py - util file for building the backbone of Faster-RCNN model
            - BoundingBox.py - called by BoundingBoxes.py, defines the data structure to store each bounding box
            - BoundingBoxes.py - called by Evaluator.py, defines the data structure to store collective bounding boxes
            - Evaluator.py - python script which contains functions that evaluate mAP for Faster-RCNN
            - faster_rcnn.py - file that stores the class for building Faster-RCNN model
            - generalized_rcnn.py - file that defines the forward network of an rcnn
            - image_list.py - file that contains classes which store the list of images and their sizes
            - nntools.py - file that contains classes which help build the experiment for storing, running, training and evaluating.
            - resnet.py - file that contains classes which store basic resnet structure
            - roi_heads.py - file that contains classes which define the head for ROI layer in RCNN.
            - rpn.py - file that contains classes which define the region proposal network
            - transform.py - file that contains classes which contain various functions for Faster-RCNN transforms
            - voc_test.py - file that contains classes which define the structure for VOC Dataset
        - utils - utility files for Faster-RCNN implementation
            - _utils_main.py - utility functions for RCNNs
            - _utils.py - utility functions for main RCNN builders
            - utils_box.py - utility functions for calculating mAP
    - Demo_FasterRCNN.ipynb - demo file for Faster-RCNN implementation. This file plots statistics, detects on test images, and generates detected video files as output. Kindly note that you need tow download checkpoint.pth.tar from the above mentioned Google Drive link and place it in checkpoint_fasterrcnn folder before running this.
    - eval_fasterrcnn.py - used to calculate and display evaluation loss and mAP on VOC Test Set.
    - Faster_RCNN_Detection.jpg - Detection example for Faster-RCNN
    - Faster_RCNN_Stats.jpg - Plotted Stats for Faster-RCNN
    - Faster_RCNN_Video.avi - Video Detected File for Faster-RCNN
    - train_fasterrcnn.py - used to perform training on Faster-RCNN

### References <br>
Apart from links above for SSD Arxiv paper and VOC dataset documentation, we referred to:
- [Project problem statement document](https://www.charles-deledalle.fr/pages/files/ucsd_ece285_mlip/projectC_object_detection.pdf)

## A project by - 

- Swapnil Aggarwal
- Utkarsh Singh
- Hitesh Sonawala
