# FIne-Grained-Car-Classification-REResNet50
CNN model based on Keras REResNet50 to classify car makes, models, and year of production

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


```
pip install git+https://github.com/qubvel/classification_models.git
```

This should install the pachage of Keras Models we use in our training

Also we choose to not include CompCars Dataset in the repo since it is a large file, thus you must download the Dataset following instructions from [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt), **place it on the top level of the repo and name the directory as "CompCars_Dataset".** Make sure you see two directories data/, sv_data/ and a README.txt by the end of this step

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

### SEResnet1.py
### SEResnet1.py

To this point, you should be able to interpret the data in [outputs/](outputs/) freely

## Authors

* **Liyang Xie** - [LY-Xie](https://github.com/LY-Xie)

* **Stefan Li** - [StefStroke](https://github.com/StefStroke)
