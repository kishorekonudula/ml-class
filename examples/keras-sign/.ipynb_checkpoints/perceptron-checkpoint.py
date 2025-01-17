# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "mae"
config.optimizer = "adam"
config.epochs = 10

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data() # y data is the corresponding alphabet number. There are only 25 because 2 alphabets are removed as they involve hand movement in the sign language

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# create model
model = Sequential()
model.add(Flatten())
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])

# The default model in this case is performing very badly. The accuracy is pretty bad.
# Normalizing the data improves the accuracy. One way to do it for this data is to divide the data by 255 Since the input is between 0 and 255
# 