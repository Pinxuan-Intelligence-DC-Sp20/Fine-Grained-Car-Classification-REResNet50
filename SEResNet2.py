# install model from Keras and 
pip install git+https://github.com/qubvel/classification_models.git

# import statements
import os

from math import *
import random
from scipy.io import loadmat
import numpy as np
import pandas as pd

import cv2
from sklearn.model_selection import train_test_split

import keras
from classification_models.keras import Classifiers
from keras.applications import imagenet_utils
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard


# dict with 'make_names' and 'model_names' as keys
mat = loadmat('car_type/make_model_name.mat')
# make_names, model_names are np arrays
make_names = mat['make_names']
model_names = mat['model_names']


# import the models we need and ImageDataGenerator for generating train_X, train_y, test_X, test_y
# ImageDataGenerator is also used for data augmentation
SEResNet50, preprocess_input = Classifiers.get('seresnet50')

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip = True)



# cancatnating make, model and year into classes
path = 'CompCars_DataSet/data/images'
classes = []
for make in os.listdir(path):
  if int(make) <= 10:
    make_path = path + '/' + make
    for model in os.listdir(make_path):
      model_path = make_path + '/' + model
      for year in os.listdir(model_path):
        class_name = str(model) + '_' + str(year)
        if class_name not in classes:
          classes.append(class_name)
num_classes = len(classes)

# %%
data = []
path = '/content/drive/My Drive/fine_grained_car_classification/CompCars_DataSet/data/images'
for make in os.listdir(path):
  if int(make) <= 10:
    make_path = path + '/' + make
    for model in os.listdir(make_path):
      model_path = make_path + '/' + model
      for year in os.listdir(model_path):
        year_path = model_path + '/' + year
        for image in os.listdir(year_path):
          im = cv2.imread(year_path + '/' + image)
          data.append((im, classes.index(str(model) + '_' + str(year))))

num_data = len(data)
num_data

# %%
random.shuffle(data)
data_train = data[0:300]
data_test = data[100:-1]
len(data_train)

# %%
def gen_train(d, max_size_length, batch_size=6):
    while 1:
        y = []
        img_list = []
        max_h, max_w = 0, 0
        img_info = random.sample(d, batch_size)
        for i in img_info:
          y.append(i[1])
          img = i[0]
          h, w, _ = img.shape
          if w > h and w > max_size_length:
              new_w = max_size_length
              new_h = h * max_size_length // w
              img = cv2.resize(img, (new_w, new_h))
              h, w, _ = img.shape
          elif h > w and h > max_size_length:
              new_h = max_size_length
              new_w = w * max_size_length // h
              img = cv2.resize(img, (new_w, new_h))
              h, w, _ = img.shape
          max_h = h if h > max_h else max_h
          max_w = w if w > max_w else max_w
          img_list.append(img)

        x = np.zeros((batch_size, max_h, max_w, 3))

        for i in range(batch_size):
            img = img_list[i]
            h, w, _ = img.shape
            transform_parameters = datagen.get_random_transform((h, w, 3), seed=None)
            x[i, 0:h, 0:w, :] = datagen.apply_transform(img, transform_parameters)

        x = preprocess_input(x)
        y = to_categorical(y, num_classes)
        yield x, y

# %%
def gen_test(d, max_size_length, batch_size=6):
    while 1:
        y = []
        img_list = []
        max_h, max_w = 0, 0
        img_info = random.sample(d, batch_size)
        for i in img_info:
          y.append(i[1])
          img = i[0]
          h, w, _ = img.shape
          if w > h and w > max_size_length:
              new_w = max_size_length
              new_h = h * max_size_length // w
              img = cv2.resize(img, (new_w, new_h))
              h, w, _ = img.shape
          elif h > w and h > max_size_length:
              new_h = max_size_length
              new_w = w * max_size_length // h
              img = cv2.resize(img, (new_w, new_h))
              h, w, _ = img.shape
          max_h = h if h > max_h else max_h
          max_w = w if w > max_w else max_w
          img_list.append(img)

        x = np.zeros((batch_size, max_h, max_w, 3))

        for i in range(batch_size):
            img = img_list[i]
            h, w, _ = img.shape
            transform_parameters = datagen.get_random_transform((h, w, 3), seed=None)
            x[i, 0:h, 0:w, :] = datagen.apply_transform(img, transform_parameters)

        x = preprocess_input(x)
        y = to_categorical(y, num_classes)
        yield x, y

# %%
model = SEResNet50(input_shape=(None,None,3), weights=None, include_top=True,classes=num_classes)
adam = Adam(lr=0.0006)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 8
val_step = ceil(len(data_train) / (2 * batch_size))
print(val_step)
train_loader = gen_train(data_train, 256, batch_size=batch_size)
test_loader = gen_test(data_train, 256, batch_size=batch_size)

checkpoint = ModelCheckpoint(filepath='./models/weights-senet-{epoch:02d}-{val_loss:.2f}.h5',
                             monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto')
# earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)


print('-----------Start training-----------')
history = model.fit_generator(train_loader,
    steps_per_epoch = 50,
    epochs = 50,
    initial_epoch = 0,
    validation_data = test_loader,
    shuffle = True,
    validation_steps = val_step,
    callbacks = [checkpoint, reduce_lr, tensorboard])

model.save('/content/drive/My Drive/fine_grained_car_classification/car_type/trial4.h5')

# %%
acc_dict = history.history
acc_dict.keys()
trial4 = pd.DataFrame(acc_dict)
trial4.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns


fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(ax=ax, data=trial4)

plt.ylim(0, 5)
plt.setp(ax.get_legend().get_texts(), fontsize='15')
plt.xlabel('# Epochs')
plt.ylabel('Training Stats')
plt.title('Training Progress')


# %%
