import tensorflow as tf
from cnn_utility import *
from utility import *

model = tf.keras.models.load_model('models/adam-hope-1568405555.h5')
model.summary()

(stairs_x, stairs_y) = load_data_by_name('stairs')
stairs_x = stairs_x.reshape([-1,120,160,3])

stairs_loss, stairs_acc = model.evaluate(stairs_x, stairs_y, batch_size=32, verbose=0)

(flat_x, flat_y) = load_data_by_name('flat')
flat_x = flat_x.reshape([-1,120,160,3])

flat_loss, flat_acc = model.evaluate(flat_x, flat_y, batch_size=32, verbose=0)

(obstacles_x, obstacles_y) = load_data_by_name('obstacles')
obstacles_x = obstacles_x.reshape([-1,120,160,3])

obstacles_loss, obstacles_acc = model.evaluate(obstacles_x, obstacles_y, batch_size=32, verbose=0)

(valid_x, valid_y) = load_data_by_name('validation')
valid_x = valid_x.reshape([-1,120,160,3])

valid_loss, valid_acc = model.evaluate(valid_x, valid_y, batch_size=32, verbose=0)

print('Val loss:', valid_loss)
print('Val accuracy:', valid_acc)

print('Stairs loss:', stairs_loss)
print('Stairs accuracy:', stairs_acc)

print('Flat loss:', flat_loss)
print('Flat accuracy:', flat_acc)

print('obstacles loss:', obstacles_loss)
print('obstacles accuracy:', obstacles_acc)


# Testing with real data
img = cv2.imread('real.jpg', cv2.COLOR_BGR2RGB) / 255.0

input_img = preproccesing_image(img)
input_img = np.zeros(input_img.shape)

arr = np.array([input_img,])
arr = arr.reshape([1,120,160,3])

real_predict = model.predict(arr)
idx = real_predict >= 0.5
real_predict[idx] = 1
idx = np.logical_not(idx)
real_predict[idx] = 0
p1, p2 = get_fov(0, 0, 0, 90, 60)
create_nodes(0, 0, p1[0], p1[1], p2[0], p2[1])

b, g, r = cv2.split(input_img)
rgb_img = cv2.merge([r, g, b])

set_labels(real_predict[0].astype(bool))
# plt.subplot(1, 2, 1)
plot_data()
legend_player, = plt.plot(0, 0, color='#000099', marker='o', label='Spieler')
legend_free, = plt.plot([], [], marker='o', color='green', mew=0.1, label='begehbar')
legend_blocked, = plt.plot([], [], marker='o', color='red', mew=0.1, label='blockiert')
plt.legend(handles=[legend_player, legend_free, legend_blocked],
           prop={'size': 12})

# plt.subplot(1, 2, 2)
# plt.imshow(rgb_img)
plt.axis('off')
plt.show()
