'''
Compiling the model and training
'''
import matplotlib.pyplot as plt

from library import *
from build_model import *


# Load data
with h5py.File('P:/PycharmProjects/pythonProject/storing/data.hdf5', 'r') as hf:
    x_train = hf['x_train'][()]
    y_train = hf['y_train'][()]
    x_valid = hf['x_valid'][()]
    y_valid = hf['y_valid'][()]
    x_test = hf['x_test'][()]
    y_test = hf['y_test'][()]


unet = build_unet()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
#opt = tf.keras.optimizers.SGD(learning_rate=0.00001)
unet.compile(optimizer=opt,
             loss="binary_crossentropy",
             metrics=['accuracy'])
unet.summary()

num_of_epochs = 30

path_checkpoint = "training_1/cp.ckpt"
callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                              save_weights_only=True,
                                              verbose=1)

history = unet.fit(x=x_train,
                   y=y_train,
                   validation_data=(x_valid, y_valid),
                   epochs=num_of_epochs,
                   shuffle=True,
                   verbose=1,
                   batch_size=8)

unet.save('unet_fullOvarian.h5')

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

