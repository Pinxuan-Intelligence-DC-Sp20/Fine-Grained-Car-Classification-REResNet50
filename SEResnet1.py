import cv2, os, random, csv
import keras
from keras.models import load_model
from keras.applications import imagenet_utils
import numpy as np
from scipy.io import loadmat
from classification_models.keras import Classifiers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
preprocess_input = lambda x: imagenet_utils.preprocess_input(x, mode='torch') #Comment if training new model
make_names = loadmat('make_model_name.mat')['make_names']
model_names = loadmat('make_model_name.mat')['model_names']

#Uncomment if training new model
#SEResNet50, preprocess_input = Classifiers.get('seresnet50')

num_classes = 4454

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_iter = csv.reader(open('new_train.csv', 'rt'))
next(train_iter)
test_iter = csv.reader(open('new_test.csv', 'rt'))
next(test_iter)

def gen_train(batch_size=6):
    while True:
        y = []
        img_list = []
        max_h, max_w = 0, 0
        for b in range(batch_size):
          try:
            i = next(train_iter)
          except:
            train_iter = csv.reader(open('new_train.csv', 'rt'))
            next(train_iter)
            i = next(train_iter)
          y.append(i[3])
          img = cv2.imread(i[1])
          h, w, _ = img.shape
          if w > h and w > 400:
              new_w = 400
              new_h = h * 400 // w
              img = cv2.resize(img, (new_w, new_h))
              h, w, _ = img.shape
          elif h > w and h > 400:
              new_h = 400
              new_w = w * 400 // h
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

def gen_test(batch_size=6):
    while True:
        y = []
        img_list = []
        max_h, max_w = 0, 0
        for b in range(batch_size):
            try:
                i = next(test_iter)
            except:
                test_iter = csv.reader(open('new_train.csv', 'rt'))
                next(test_iter)
                i = next(test_iter)
            y.append(i[3])
            img = cv2.imread(i[1])
            h, w, _ = img.shape
            if w > h and w > 400:
                new_w = 400
                new_h = h * 400 // w
                img = cv2.resize(img, (new_w, new_h))
                h, w, _ = img.shape
            elif h > w and h > 400:
                new_h = 400
                new_w = w * 400 // h
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

#Continue training
model = load_model('trial5.h5')

#Training new model
#model = SEResNet50(input_shape=(None,None,3), include_top=True,classes=num_classes)


adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 8
train_loader = gen_train(batch_size=batch_size)
test_loader = gen_test(batch_size=batch_size)

checkpoint = ModelCheckpoint(filepath='./weights-senet-{epoch:02d}-{val_loss:.2f}.h5',
                             monitor='val_loss', save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto')

earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

print('-----------Start training-----------')
model.fit_generator(train_loader,
    steps_per_epoch = 16000,
    epochs = 10,
    initial_epoch = 0,
    validation_data = test_loader,
    validation_steps = 4000,
    callbacks = [checkpoint, earlystop, reduce_lr])

'''
train_img = train_set.readline()[0:-1]
count = 0
X = []
y = []
classes = []
while train_img:
    _ = train_img.split('/')
    name = str(_[0]) + '_' + str(_[1]) + '_' + str(_[2])
    if name not in classes:
        classes.append(name)
        y.append(len(classes))
    else:
        y.append(classes.index(name))
    im = cv2.imread('D:/SAAS/DC/project/fine-grained vehicle classification/data/data/image/' + train_img)
    h, w, _ = im.shape
    if w > h and w > 600:
        new_w = 600
        new_h = h * 600 // w
        im = cv2.resize(im, (new_w, new_h))
    elif h > w and h > 600:
        new_h = 600
        new_w = w * 600 // h
        img = cv2.resize(im, (new_w, new_h))
    im = preprocess_input(im)
    X.append(im)
    train_img = train_set.readline()[0:-1]
    count += 1
    if count % 100 == 1:
        print(count)
X = np.asarray(X)
y = np.asarray(y)

base_model = load_model('models/final.h5', compile=False)
model = keras.models.Sequential()
model.add(base_model)
model.add(keras.layers.Dense(3, activation="softmax"))
model.compile(optimizer="SGD",loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y)
'''

model.save('trial5.h5')
