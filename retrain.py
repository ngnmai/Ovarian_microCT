from library import *

with h5py.File('P:/PycharmProjects/pythonProject/storing/data.hdf5', 'r') as hf:
    x_train = hf['x_train'][()]
    y_train = hf['y_train'][()]
    x_test = hf['x_test'][()]
    y_test = hf['y_test'][()]

unet_model = tf.keras.models.load_model('P:/PycharmProjects/pythonProject/storing/unet_lung.h5')
#unet_model = tf.keras.models.load_model('P:/PycharmProjects/pythonProject/test_unet.h5')
unet_model.summary()

num_of_epochs = 5
history = unet_model.fit(x=x_train,
                   y=y_train,
                   epochs=num_of_epochs,
                   shuffle=True,
                   verbose=1,
                   batch_size=1)

unet_model.save('P:/PycharmProjects/pythonProject/storing/unet_lungAndOvarian.h5')

plt.plot(history.history['loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()

# TESTING
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
#plt.imshow(x_test[pic_1st][...,0], cmap='bone')
plt.imshow(y_test[pic_1st][...,0], cmap='Purples', alpha=1)
plt.title('original mask')

plt.subplot(3, 3, 3)
#plt.imshow(x_test[pic_1st][...,0], cmap='bone')
plt.imshow(y_pred[pic_1st][...,0], cmap='Purples', alpha=1)
plt.title('predicted mask')


# Second testing sample
plt.subplot(3, 3, 4)
plt.imshow(x_test[pic_2nd][...,0], cmap='bone')
plt.title('original CT image')

plt.subplot(3, 3, 5)
#plt.imshow(x_test[pic_2nd][...,0], cmap='bone')
plt.imshow(y_test[pic_2nd][...,0], cmap='Purples', alpha=1)
plt.title('original mask')

plt.subplot(3, 3, 6)
#plt.imshow(x_test[pic_2nd][...,0], cmap='bone')
plt.imshow(y_pred[pic_2nd][...,0], cmap='Purples', alpha=1)
plt.title('predicted mask')


# Third testing sample
plt.subplot(3, 3, 7)
plt.imshow(x_test[pic_3rd][...,0], cmap='bone')
plt.title('original CT image')

plt.subplot(3, 3, 8)
#plt.imshow(x_test[pic_3rd][...,0], cmap='bone')
plt.imshow(y_test[pic_3rd][...,0], cmap='Purples', alpha=1)
plt.title('original mask')

plt.subplot(3, 3, 9)
#plt.imshow(x_test[pic_3rd][...,0], cmap='bone')
plt.imshow(y_pred[pic_3rd][...,0], cmap='Purples', alpha=1)
plt.title('predicted mask')

plt.show()