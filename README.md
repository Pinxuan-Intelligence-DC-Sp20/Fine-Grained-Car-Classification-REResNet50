# Fine-Grained-Car-Classification-REResNet50
CNN model based on Keras REResNet50 to classify car makes, models, and year of production

## Problem Statement
### Task:  
come up with a model that, given a random image of a car, can predict the car’s make, model, and year with 95% accuracy. 
### Dataset: 
We utilize the CompCars Dataset by Cornell University, a dataset of over 160,000 HD images and surveillance images with labels to train on. The HD images are mostly from the web, whereas the surveillance images are taken by actual surveillance cameras. As for the labels, there are over 160 makes. Within each make, there are 1-15 models. And within each model, there are about 10 release years, accounting up to more than 11000 classes
Significance: 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


```
pip install git+https://github.com/qubvel/classification_models.git
```

This should install the pachage of Keras Models we use in our training

Also we choose to not include CompCars Dataset in the repo since it is a large file. Thus you must download the Dataset following instructions from [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt), then **place it on the top level of the repo and name the directory as "CompCars_Dataset".** Make sure you see two directories data/, sv_data/ and a README.txt by the end of this step

## Prerequisites

Interpreter you need to deploy

```
Python 3.7.0 or higher
```

Packages and libraries you need to install

```
pip/pip3
Python 3.7.0 or higher

csv
scipy.io
numpy
pandas

cv2
sklearn.model_selection.import train_test_split

keras
classification_models.keras.Classifiers
keras.applications.imagenet_utils
keras.models.load_model
keras.utils.to_categorical
keras.preprocessing.image.ImageDataGenerator
keras.optimizers.Adam
keras.callbacks.ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard

matplotlib.pyplot
seaborn
```

## Running the Code

The "meat" of our model is in [SEResnet1.py](SEResnet1.py) file and [SEResNet2.py](SEResNet2.py). These two files are two slightly different implementation of the SEResnet, in particular with different hyperparameters and data agumentation schemes

## Train test split.py
[train_test_split.py](train_test_split.py) on the top level of the repo is specifically used for [SEResnet1.py](SEResnet1.py). **You must run [train_test_split.py](train_test_split.py) before running [SEResnet1.py](SEResnet1.py),** generating two .csv files as a result. The Above mentioned step is not necessary for [SEResNet2.py](SEResNet2.py), which has its train_test_split process embedded in the .py file.

## SEResNet2.py
There are various hyperparameters well documented in [SEResNet2.py](SEResNet2.py). In comments starting with 
```
#TODO:
```
you will see the meaning of these hyperparameters and the interval, between which you can tune them. Feel free to use our model for your own purposes

## SEResNet2.py
There are various hyperparameters well documented in [SEResNet2.py](SEResNet2.py). In comments starting with 
```
#TODO:
```

## Authors

* **Liyang Xie** - [LY-Xie](https://github.com/LY-Xie)

* **Stefan Li** - [StefStroke](https://github.com/StefStroke)
