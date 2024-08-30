# Challenge AppSec Biometrics 

This repository contains resources for training and evaluating a liveness detection model. 
The primary components include Jupyter notebooks for experimentation, model files, and a script for  webcam integration.


## Directory Structure

- `attack_msu.ipynb`: Jupyter notebook for evaluating attacks on the MSU dataset.
- `train_msu_cat.ipynb`: Jupyter notebook for training the VGG19 model on the MSU dataset.
- `adv_train_msu.ipynb`: Jupyter notebook for training with adversarial examples on the MSU dataset.
- `webcam_msu.py`: Script for running liveness detection using a webcam.
- `webcam_msu_robust.py`: Script for running liveness detection with the robust model using a webcam.
- `dataset`: Directory containing the MSU MFSD dataset for liveness detection.
- `trainHistory`: Training history logs for the models.
- `adversarial_batches`: Directory containing adversarially perturbed images for training and evaluation.
- `attacks`: Directory containing different types of adversarial attacks used in experiments.
- `face_detector`: Code and resources for detecting faces in images.
- `requirements.txt`: Python package dependencies required for the project.
- `vgg19_msu_categorical.keras`: Keras model file for VGG19 trained on the MSU dataset with categorical cross-entropy.
- `vgg19_msu_categorical_robust.keras`: Keras model file for a robust VGG19 model trained with adversarial defenses.
