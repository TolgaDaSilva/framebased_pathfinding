# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
from cnn_utility import *

NAME = f"cnn-model-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f'.\\logs\\{NAME}')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# split_data()

(train_images, train_labels), (test_images, test_labels) = load_data()

(validation_x, validation_y) = load_validation()
validation_x = validation_x.reshape(validation_x.shape[0],120,160,3)
# validation_x = np.zeros(validation_x.shape)

# dont need train and test split, because of validation data
train_images = np.concatenate((train_images, test_images))
train_labels = np.concatenate((train_labels, test_labels))

train_images = train_images.reshape([-1,120,160,3])
train_images = np.zeros(train_images.shape)

test_images = test_images.reshape([-1,120,160,3])

input_shape = (120,160,3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, data_format="channels_last"))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(160, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='sigmoid'))

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['binary_accuracy'])

history = model.fit(train_images, train_labels, batch_size=32, epochs=21, verbose=1,
                    validation_data=(validation_x, validation_y), callbacks=[tensorboard])

model.save(f'models/{NAME}.h5')
with open(f'history/{NAME}', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
(stairs_x, stairs_y) = load_data_by_name('stairs')
stairs_x = stairs_x.reshape([-1,120,160,3])

stairs_loss, stairs_acc = model.evaluate(stairs_x, stairs_y, verbose=0)

(flat_x, flat_y) = load_data_by_name('flat')
flat_x = flat_x.reshape([-1,120,160,3])

flat_loss, flat_acc = model.evaluate(flat_x, flat_y, verbose=0)

(obstacles_x, obstacles_y) = load_data_by_name('obstacles')
obstacles_x = obstacles_x.reshape([-1,120,160,3])

obstacles_loss, obstacles_acc = model.evaluate(obstacles_x, obstacles_y, verbose=0)

(valid_stairs_x, valid_stairs_y) = load_data_by_name('unknown_stairs')
valid_stairs_x = valid_stairs_x.reshape([-1,120,160,3])

valid_loss, valid_acc = model.evaluate(valid_stairs_x, valid_stairs_y, verbose=0)

print('Unknown stairs loss:', valid_loss)
print('Unknown stairs accuracy:', valid_acc)

print('Stairs loss:', stairs_loss)
print('Stairs accuracy:', stairs_acc)

print('Flat loss:', flat_loss)
print('Flat accuracy:', flat_acc)

print('Obstacles loss:', obstacles_loss)
print('Obstacles accuracy:', obstacles_acc)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()