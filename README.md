# SOTA-DL-EchoSegmentation
Evaluation of state-of-the-art (SOTA) deep learning (DL) models in the segmentation of left and right ventricles in parasternal short-axis echocardiograms (PSAX-echo)
SIUE Biomedical Imaging Research Lab - BIRL
..:: https://siue-biomedicalimaginglab.com/ ::..

A deep learning based approach to segment medical echocardiography PSAX images into its 3 main heart structures:
* Left chamber
* Left myocardium
* Right ventricle

Based on 
https://github.com/raventan95/echo-plax-segmentation


## Introduction

Echocardiography images (echo) are ultrasound scans of the heart. Here, a deep learning (DLM) that has its architecture based on the U-Net RESNET is trained to segment the echos. The parasternal short axix (PSAX) view is considered. In this particular view, 3 main heart structures can be visible, the right ventricle, left ventricle, and left myocardium. This segmentation is our second step in our research focused in the detection and quantification of fat around the heart using echocardiograms.

The DLM is built using TensorFlow Keras library, with a U-Net RESNET architecture. The input is and IQ file containing the radio frequency (RF) data of the echocardiographic image, then it is normalized and reshaped to be passed as input of the DLM, the output is passed through a softmax layer to predict 1 out of 4 segmentation objects (3 heart structures + background). It has managed to achieve a dissimilarity coefficient (DSC) with and accuracy of 75.3% on average. In total a dataset of 1737 images were used with a training/validation/test split of 1213/174/350 images respectively.

## Dependencies

This Python file was tested on:
- Python 3.7
- ImageIO 2.9.0
- imageio-ffmpeg 0.5.1
- Keras 2.3.1
- MatPlotLib 3.3.2
- NumPy 1.19.2
- SciKit-Image 0.17.2

## How to run

1. Create a system environment using **_Anaconda 3_**.
2. Create a `Fork` from this repository to your GitHub.
3. `Clone` the repo with the PyCharm tools.
4. Download a trained model weights from [OneDrive/Training_Models](https://siuecougars.sharepoint.com/:f:/r/sites/cardiacfatsegmentation/Shared%20Documents/Training_Models/2D-Echo-PSAX-Segmentation?csf=1&web=1&e=689kmf) to the folder `model/` in the local directory.
5. Select the IQ file that you want to segment and create a video. IQ files are in [OneDrive/UltrasoundData](https://siuecougars.sharepoint.com/:f:/r/sites/cardiacfatsegmentation/Shared%20Documents/UltrasoundData?csf=1&web=1&e=mpW4Zo).
6. Run the Python script
```
python main.py
```
5. Segmented videos are saved in `result/` folder.

## Sample results

Here are two examples of segmented videos from the model, evaluated on selected IQ files. Sample output videos are in [OneDrive/VideosAndImages/EchoPSAX Segmentation ResNet](https://siuecougars.sharepoint.com/:f:/r/sites/cardiacfatsegmentation/Shared%20Documents/VideosAndImages/EchoPSAX%20Segmentation%20ResNet?csf=1&web=1&e=lftE7P)

- red : left ventricle
- purple : left myocardium
- yellow : right ventricle

| ![](sample/MF0308PRE_5_seg.gif) |  ![](sample/MF0519_10_seg.gif) |
|:--:|:--:|
| MF0308PRE_5 segmented | MF0519_10 segmented |
