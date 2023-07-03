'''
Compiling the model and training
'''
import matplotlib.pyplot as plt

from library import *
from build_model import *


# Load data
with h5py.File('P:/PycharmPorjects/pythonProject/storing/data.hdf5', 'r') as hf:
    x_train = hf['x_train'][()]
    y_train = hf['y_train'][()]
    x_test = hf['x_test'][()]
    y_test = hf['y_test'][()]


unet = build_unet()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
#opt = SGD(learning_rate=0.00001)
unet.compile(optimizer= opt,
             loss="binary_crossentropy",
             metrics=['accuracy'])
unet.summary()

num_of_epochs = 20

path_checkpoint = "training_1/cp.ckpt"
callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                              save_weights_only=True,
                                              verbose=1)

history = unet.fit(x=x_train,
                   y=y_train,
                   epochs=num_of_epochs,
                   shuffle=True,
                   verbose=1,
                   batch_size=16)

unet.save('test_unet.h5')

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

