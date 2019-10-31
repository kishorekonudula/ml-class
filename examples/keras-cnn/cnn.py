import tensorflow as tf
import wandb

# initialize wandb & set hyperparamers
run = wandb.init()
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dropout = 0.4
config.dense_layer_size = 256
config.img_width = 28
config.img_height = 28
config.epochs = 100
config.learn_rate_low = 0.001

# load and normalize data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels = [str(i) for i in range(10)]

# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128,
                                 (config.first_layer_conv_width,
                                  config.first_layer_conv_height),
                                 input_shape=(28, 28, 1),
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(config.dropout))
model.add(tf.keras.layers.Conv2D(128,
                                 (config.first_layer_conv_width,
                                  config.first_layer_conv_height),
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(config.dropout))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(config.dense_layer_size, activation='selu'))
model.add(tf.keras.layers.Dropout(config.dropout))
model.add(tf.keras.layers.Dense(config.dense_layer_size, activation='selu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=config.learn_rate_low),
              metrics=['accuracy'], weighted_metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[wandb.keras.WandbCallback(data_type="image", save_model=False), tf.keras.callbacks.ReduceLROnPlateau(), tf.keras.callbacks.EarlyStopping(patience=7)])

# , beta_1=0.9, beta_2=0.999, amsgrad=False
# To change the initializer
# model.add(Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal',
#                  input_shape=(config.img_width, config.img_height, 1)))