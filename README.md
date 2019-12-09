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
- Faster-RCNN: Download [Faster-RCNN (16)](https://drive.google.com/open?id=1OahGZd_7ocgdDPWWCyBsqwCD0S3m0DIS) and [Faster-RCNN (32)](https://drive.google.com/open?id=1LH24iGxJt-Luzkm5p5QxnKRD-D2sPZTT) checkpoint files and put the 'checkpoint.pth.tar' file in respective folder 'checkpoint_faster_rcnn' or 'checkpoint_fasterrcnn32'.

- SSD: Download from 'https://bit.ly/2YCJMks' and put the 'BEST_checkpoint_ssd300.pth.tar' in the data folder.

### Datasets
[2012 version](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) of Pascal VOC dataset - well known dataset for object detection/classification/segmentation. Contains 100k images for training and validation containing bounding boxes with 20 categories of objects.<br />
[2007 Trainval version](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) of Pascal VOC2007 train dataset.<br />
[2007 Test version](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) of Pascal VOC2007 test dataset.

### Demo
#### Faster-RCNN
- Run **Demo_FasterRCNN.ipynb** notebook to run Faster-RCNN (16) to plot statistics, detect images and videos on PascalVOC2007 dataset (In the notebook, set download=False for dataset loader if you want to use your custom dataset, and give the appropriate path to dataset loader)
- Run **Demo_FasterRCNN32.ipynb** notebook to run Faster-RCNN (32) to plot statistics, detect images and videos on PascalVOC2007 dataset (In the notebook, set download=False for dataset loader if you want to use your custom dataset, and give the appropriate path to dataset loader)

##### Training
Run **train_fasterrcnn.py** or **train_fasterrcnn32.py** script file to train the Faster-RCNN model (16) or Faster-RCNN model (32) on the PascalVOC2012 dataset.

##### Evaluation
Run **eval_fasterrcnn** or **eval_fasterrcnn32**notebook to evaluate the respective Faster-RCNN models on the PASCALVOC2007 test set.

#### SSD
- Run **SSD_Demo_IMG.ipynb** notebook to run Single-Shot Detection on a random image from the PascalVOC2007 dataset.
### Training
Before you begin, make sure to save the required data files for training and validation. To do this, run the contents of **create_data_lists.py** after pointing it to the VOC2007 and VOC2012 folders in your downloaded data.<br />
Run **ssd_train.py** to train the SSD model on the PascalVOC2012 dataset.<br />
To resume training at a checkpoint, point to the corresponding file with the checkpoint parameter at the beginning of the code.
### Evaluation
Run **ssd_eval.py** notebook to evaluate the SSD model on the PascalVOC2007 test set.
### Performance <br>
On [UCSD Data Science and Machine Learning Cluster](https://datahub.ucsd.edu/hub/home):

| ------------- | Faster-RCNN(32)  | Faster-RCNN(16) | SSD |
| ------------- | ---------------- | --------------- | --- |
| Training      |  60%             | 76%             | 72%   | 
| Evaluation    |  46%             | 63%             | 70%   |

### Directory structure

- faster_rcnn/ - files for Faster-RCNN implementation:
    - checkpoint_fasterrcnn/ - folder that stores the checkpoint files
        - config.txt - the config file for the experiment with batch size 16
        - checkpoint.pth.tar - download this file from the above mentioned Google Drive link and place here
    - checkpoint_fasterrcnn32/ - folder that stores the checkpoint files
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
    - Demo_FasterRCNN.ipynb - demo file for Faster-RCNN (16) implementation. This file plots statistics, detects on test images, and generates detected video files as output. Kindly note that you need to download checkpoint.pth.tar from the above mentioned Google Drive link and place it in checkpoint_fasterrcnn folder before running this.
    - Demo_FasterRCNN32.ipynb - demo file for Faster-RCNN (32) implementation. This file plots statistics, detects on test images, and generates detected video files as output. Kindly note that you need to download checkpoint.pth.tar from the above mentioned Google Drive link and place it in checkpoint_fasterrcnn folder before running this.
    - eval_fasterrcnn.py - used to calculate and display evaluation loss and mAP on VOC Test Set for Faster-RCNN (16).
    - eval_fasterrcnn32.py - used to calculate and display evaluation loss and mAP on VOC Test Set for Faster-RCNN (32).
    - Faster_RCNN_Detection.jpg - Detection example for Faster-RCNN (16)
    - Faster_RCNN_Stats.jpg - Plotted Stats for Faster-RCNN (16)
    - Faster_RCNN_Video.avi - Video Detected File for Faster-RCNN (16)
    - Faster_RCNN_Detection32.jpg - Detection example for Faster-RCNN (32)
    - Faster_RCNN_Stats32.jpg - Plotted Stats for Faster-RCNN (32)
    - Faster_RCNN_Video32.avi - Video Detected File for Faster-RCNN (32)
    - train_fasterrcnn.py - used to perform training on Faster-RCNN (16)
    - train_fasterrcnn32.py - used to perform training on Faster-RCNN (32)
    
- SSD/ - files for SSD implementation:
    - Video_experiments/ - folder that contains video test and detected file
        - SSD_Video_Test.mp4 - Test file
        - SSD_Video.avi - Object detection on video
    - experiment/ - folder that contains training plot results and dectectd images
        - Plot_loss.ipynb - notebook to plot varoius training loss for SSD
        - img - folder containg Detected File for SSD
        - Plots - folder containg Plotted Stats for SSD
    - data/ - folder that stores the checkpoint files
        - label_map.json - Label Map
        - Test_image.json - Test Images
        - Test_objects.json - Test Objects
        - Train_image.json - Train Images
        - Train_objects.json - Train Objects
        - BEST_checkpoint_ssd300.pth.tar - download this file from the above mentioned Google Drive link and place here
    - SSD_Demo_IMG.ipynb - demo file for SSD implementation. This file detects objects on test images and plots the output. Kindly note that you need to download BEST_checkpoint_ssd300.pth.tar from the above mentioned Google Drive link and place it in data directory before running this.
    - SSD_Demo_Video.ipynb - demo file for SSD implementation. This file generates detected video files as output. Kindly note that you need to download BEST_checkpoint_ssd300.pth.tar from the above mentioned Google Drive link and place it in data directory before running this.
    - create_data_list.py - Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    - detect.py - Detect objects in an image with a trained SSD300, and visualize the results.
    - eval.py - used to calculate and display evaluation loss and mAP on VOC Test Set.
    - model.py - file that stores the class for building SSD model
    - train.py - script to train the model
    - utils.py - utility functions for main SSD builders

### References <br>
Apart from links above for SSD Arxiv paper and VOC dataset documentation, we referred to:
- [Project problem statement document](https://www.charles-deledalle.fr/pages/files/ucsd_ece285_mlip/projectC_object_detection.pdf)

## A project by - 

- Swapnil Aggarwal
- Utkarsh Singh
- Hitesh Sonawala
