'''
Load trainined model and test on test data
PLotting loss function

'''
from library import *

unet_model = tf.keras.models.load_model('P:/PycharmProjects/pythonProject/microCT_project/test_unet.h5')
#unet_model = tf.keras.models.load_model('P:/PycharmProjects/pythonProject/test_unet.h5')
unet_model.summary()

with h5py.File('data.hdf5', 'r') as hf:
    x_test = hf['x_test'][()]
    y_test = hf['y_test'][()]


score = unet_model.evaluate(x=x_test, y=y_test, batch_size=16, verbose=1)
print(score)

y_pred = unet_model.predict(x_test)
fig = plt.figure()

# Pick random images for every time test
list_index = []
while True:
    num = random.randint(0, y_test.shape[0])
    if num not in list_index:
        list_index.append(num)
    if len(list_index) == 3:
        break

pic_1st = list_index[0]
pic_2nd = list_index[1]
pic_3rd = list_index[2]

#First testing sample
plt.subplot(3, 3, 1)
plt.imshow(x_test[pic_1st][...,0], cmap='bone')
plt.title('original CT image')

plt.subplot(3, 3, 2)
plt.imshow(x_test[pic_1st][...,0], cmap='bone')
plt.imshow(y_test[pic_1st][...,0], cmap='nipy_spectral', alpha=0.5)
plt.title('original mask')

plt.subplot(3, 3, 3)
plt.imshow(x_test[pic_1st][...,0], cmap='bone')
plt.imshow(y_pred[pic_1st][...,0], cmap='nipy_spectral', alpha=0.5)
plt.title('predicted mask')


# Second testing sample
plt.subplot(3, 3, 4)
plt.imshow(x_test[pic_2nd][...,0], cmap='bone')
plt.title('original CT image')

plt.subplot(3, 3, 5)
plt.imshow(x_test[pic_2nd][...,0], cmap='bone')
plt.imshow(y_test[pic_2nd][...,0], cmap='nipy_spectral', alpha=0.5)
plt.title('original mask')

plt.subplot(3, 3, 6)
plt.imshow(x_test[pic_2nd][...,0], cmap='bone')
plt.imshow(y_pred[pic_2nd][...,0], cmap='nipy_spectral', alpha=0.5)
plt.title('predicted mask')


# Third testing sample
plt.subplot(3, 3, 7)
plt.imshow(x_test[pic_3rd][...,0], cmap='bone')
plt.title('original CT image')

plt.subplot(3, 3, 8)
plt.imshow(x_test[pic_3rd][...,0], cmap='bone')
plt.imshow(y_test[pic_3rd][...,0], cmap='nipy_spectral', alpha=0.5)
plt.title('original mask')

plt.subplot(3, 3, 9)
plt.imshow(x_test[pic_3rd][...,0], cmap='bone')
plt.imshow(y_pred[pic_3rd][...,0], cmap='nipy_spectral', alpha=0.5)
plt.title('predicted mask')

plt.show()