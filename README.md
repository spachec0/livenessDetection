# Challenge AppSec Biometrics 

This repository contains resources for training and evaluating a liveness detection model. 
The primary components include Jupyter notebooks for experimentation, model files, and a script for  webcam integration.

[Report for the challenge](https://github.com/spachec0/livenessDetection/blob/master/TC_AppSec_Biometrics_report.pdf)

## Directory Structure

- `attack_msu.ipynb`: Jupyter notebook for evaluating attacks on the MSU dataset.
- `train_msu_cat.ipynb`: Jupyter notebook for training the VGG19 model on the MSU dataset.
- `adv_train_msu.ipynb`: Jupyter notebook for training with adversarial examples on the MSU dataset.
- `webcam_msu.py`: Script for running liveness detection using a webcam.
- `dataset`: Directory containing the MSU MFSD dataset for liveness detection.
- `trainHistory`: Training history logs for the models.
- `adversarial_batches`: Directory containing adversarially perturbed images for training and evaluation.
- `attacks`: Directory containing different types of adversarial attacks used in experiments.
- `face_detector`: Code and resources for detecting faces in images.
- `requirements.txt`: Python package dependencies required for the project.
- `vgg19_msu_categorical.keras`: Keras model file for VGG19 trained on the MSU dataset.
- `vgg19_msu_categorical_robust.keras`: Keras model file for a robust VGG19 model trained with adversarial defenses (adversarial training).
- `TC_AppSec_Biometrics_report.pdf`: Report for the challenge.
- `images`: Util images 


## Usage 


### To test model

```
git clone https://github.com/spachec0/livenessDetection
```
Change to downloaded directory

```
pip install -r requirements.txt
```

To test liveness model with webcam

```
python webcam_msu.py -m vgg19_msu_categorical.keras
```

To test robustness liveness model with webcam

```
python webcam_msu.py -m vgg19_msu_categorical_robust.keras
```


![alt text](https://raw.githubusercontent.com/spachec0/livenessDetection/master/images/real.png)

![alt text](https://raw.githubusercontent.com/spachec0/livenessDetection/master/images/spoof.png)


## Download dataset used in notebooks

To use the dataset in notebookes, download dataset from the following url and put into dataset/data folder: 

- https://drive.google.com/drive/folders/10dJkTgo8oozNsJRainJt4hULyqTt8J5L

