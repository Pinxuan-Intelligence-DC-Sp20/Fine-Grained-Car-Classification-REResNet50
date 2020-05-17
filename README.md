# Fine-Grained-Car-Classification-REResNet50
CNN model based on Keras REResNet50 to classify car makes, models, and year of production

## Problem Statement
### Task
come up with a model that, given a random image of a car, can predict the car’s make, model, and year with 95% accuracy. 
### Dataset
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

By this point you should be able to run the models and interpret the outcomes freely. The content below document some of our approaches considered while training the model. 

## Solution and Approach

### Model Selection
We imported the SEResNet model from the github repo https://github.com/qubvel/classification_models. It has a collection of different versions of SENet that are ready to use. We chose the SEResNet50 model (we tried several others which have about the same performance), because it is simpler and easier to train.
### Initial Results
The initial results on a simple dataset with 100 images from 17 classes were really good. We had over 98% accuracy in the prediction of model_year classes. But as it turned out, we had an erroneous train test split method that does random sampling and exposes the test set during training. 
### Optimization Process
We fixed that by shuffling the whole dataset and then calling the built in train_test_split function in sklearn. However, validation accuracy dropped drastically and converged to about 40% for SEResnet1, and 25% for SEResNet2. 


We then tried fine tuning the model by changing the reduction ratio for the SE blocks, changing the aggregation method in the SE blocks from global average pooling to global max pooling, and switching selective layers of activation function from Relu to Sigmoid. All changes do not result in a significant change in the performance


We reverted back to the original setup and tried various combinations of batch size and learning rate. We found that we get over 80% training and validation accuracy on that small dataset within 10 epochs, using learning rate 0.0005/0.0006 and batch size 16. The learning rate seems optimal because both lower learning rates (e.g. 0.0001) and higher ones (e.g. 0.001) result in convergence to a validation accuracy in the range from 40% to 70% after 25~30 epochs. The last thing that boosted the performance is resizing the images and keeping the scale (if the longer side is over 600, make it 600 and resize the shorter side proportionally), instead of simply casting them to squares of 224*224 for SEResnet1, and 256 * 256 for SEResNet2. This enabled the model to reach 100% training and validation accuracy on the small dataset after 10 epochs. 


We decided to try that on the whole dataset using the client’s server which has a usable GPU. However, the GPU on the server has limited memory and we have to reduce the batch size to 8 and resize the image so that the longer side is at most 400 instead of 600, which works on the small dataset as well. 


Currently we are still training our model, the current training and validation accuracy is 50% and 48% after 70 epochs. The improvement rate is about 4% every 10 epochs and slowly decreasing. We are not sure if the validation accuracy will ever reach over 95%, but we believe it is worth it to continue training and wait until improvement stops. 


## Authors

* **Liyang Xie** - [LY-Xie](https://github.com/LY-Xie)

* **Stefan Li** - [StefStroke](https://github.com/StefStroke)
