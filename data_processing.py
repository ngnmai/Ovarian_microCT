'''
- Processing all the data files into usable file types
- Divided data into batches
- Augmenting data if needed
'''
# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np

from library import *
from preprocessing import *

# PROCESSING ALL THE DATA FILES
# Iterating through all the data file paths
ctScan = []
maskScan = []

CT = []
mask = []
'''
ctScanPath = 'P:/StudentDocuments/Documents/TESTING DATA/ct_scans'
infectionPath = 'P:/StudentDocuments/Documents/TESTING DATA/infection_mask'
preprocess_x = 'P:/StudentDocuments/Documents/TESTING DATA/ct_scans/preprocessed_x'
preprocess_y = 'P:/StudentDocuments/Documents/TESTING DATA/infection_mask/preprocessed_y'
resized_x = 'P:/StudentDocuments/Documents/TESTING DATA/ct_scans/resized_x'
resized_y = 'P:/StudentDocuments/Documents/TESTING DATA/infection_mask/resized_y'
'''

ctScanPath = 'E:/Ovarian/training/training_images'
labelPath = 'E:/Ovarian/training/training_labels'
preprocess_x = 'E:/Ovarian/training/training_images/preprocess_x'
preprocess_y = 'E:/Ovarian/training/training_labels/preprocessed_y'
ext = '.nii'  # selective file type

resize_size = 512
# train/valid/test = 70%/15%/15%
train_size = 0.7
test_size = 0.5

# size of the images that we want to resize to
img_size = 512

# OVARIAN
images_path = ['01032022', '01092022_AreaC', '01092022_AreaD', '02092022', '04022021', '05022021']
labels_path = ['01032022_seg', '01092022_areaC_seg', '01092022_areaD_seg', '02092022_seg', '04022021_seg',
               '05022021_seg']
start_index = [29, 499, 99, 419, 749, 319]
amount_toTrim = [20, 15, 10, 15, 15, 10]
axis_ = ['x', 'x', 'y', 'y', 'z', 'z']

for index in range(len(images_path)):
    path = images_path[index] + ext
    full_path_x = ctScanPath + '/' + path
    path_y = labels_path[index] + ext
    full_path_y = labelPath + '/' + path_y
    startI = start_index[index]
    amount = amount_toTrim[index]
    CT = preProcessing(filePath=full_path_x,
                       outputPath=preprocess_x,
                       file_name=path,
                       list_=CT, axis_flag=axis_[index],
                       datatype='float64',
                       first_index=startI, amount_of_slices=amount)
    mask = preProcessing(filePath=full_path_y,
                         outputPath=preprocess_y,
                         file_name=path_y,
                         list_=mask, axis_flag=axis_[index],
                         datatype='float64',
                         first_index=startI, amount_of_slices=amount)
    ctScan.append(full_path_x)
    maskScan.append(full_path_y)
    print(full_path_x)

# Looping through filepaths
'''

# LUNG
for path in os.listdir(ctScanPath):
    if path.endswith(ext):
        fullpath = ctScanPath + '/' + path
        print(fullpath)
        CT = preProcessing(fullpath, preprocess_x, path, CT, True, datatype='float64')
        ctScan.append(fullpath)
print("x_train ", len(ctScan))

for path in os.listdir(labelPath):
    if path.endswith(ext):
        fullpath = labelPath + '/' + path
        print(fullpath)
        mask = preProcessing(fullpath, preprocess_y, path, mask, False, datatype='int')
        maskScan.append(fullpath)
print("y_train ", len(maskScan))

'''
# AUGMENTING MORE DATA IMAGES
CT = np.array(CT)
mask = np.array(mask)
CT_original = CT
mask_original = mask
CT = np.squeeze(CT)
CT = np.reshape(CT, (85, 262144))
mask = np.squeeze(mask)
mask = np.reshape(mask, (85, 262144))
amount = CT_original.shape[0]
for i in range(CT.shape[0]):
    img = CT_original[i]
    label = mask_original[i]
    img = np.squeeze(img)

    # augment by rotating
    img_rotate = rotate(img, 90)
    mask_rotate = rotate(label, 90)

    # augment by flipping vertically
    img_flip = flip_vertically(img, True)
    mask_flip = flip_vertically(label, True)

    # augment adding noise
    img_noise = add_noise(img)
    mask_noise = label

    # reshape to add them to the matrices
    img_rotate = np.reshape(img_rotate, (1, 262144))
    mask_rotate = np.reshape(mask_rotate, (1, 262144))
    img_flip = np.reshape(img_flip, (1, 262144))
    mask_flip = np.reshape(mask_flip, (1, 262144))
    img_noise = np.reshape(img_noise, (1, 262144))
    mask_noise = np.reshape(mask_noise, (1, 262144))

    # adding all into the matrices
    CT = np.append(CT, img_rotate, axis=0)
    mask = np.append(mask, mask_rotate, axis=0)
    amount += 1

    CT = np.append(CT, img_flip, axis=0)
    mask = np.append(mask, mask_flip, axis=0)
    amount += 1

    CT = np.append(CT, img_noise, axis=0)
    mask = np.append(mask, mask_noise, axis=0)
    amount += 1

print("CHECKING SHAPE OF DATA")
CT = np.reshape(CT, (amount, 512, 512))
mask = np.reshape(mask, (amount, 512, 512))
print(CT_original.shape)
print(CT.shape)
CT = CT[..., np.newaxis]
mask = mask[..., np.newaxis]
print(CT.shape)
print(mask.shape)

# NORMALIZING DATA
CT_255 = []
mask_255 = []

for i in range(CT.shape[0]):
    img = CT[i][..., 0]
    label_255 = mask[i][...,0]
    img_255 = adjust(img)
    label_255 = adjust(label_255)
    #img_255 = inverse_log_transform(img_255)
    CT_255.append(img_255)
    mask_255.append(label_255)

CT_255 = np.array(CT_255).astype('int')
mask_255 = np.array(mask_255).astype('int')
CT_255 = CT_255[..., np.newaxis]
mask_255 = mask_255[..., np.newaxis]
print('**********')
print(CT_255.shape)
print(mask_255.shape)

'''

for i in range(CT_255.shape[0]):
    print_img = CT_255[i][..., 0]
    if i % 1 == 0:
        plt.subplot(1, 2, 1)
        plt.imshow(print_img, cmap='bone')
        plt.title('adjust to 255')
        plt.subplot(1, 2, 2)
        plt.imshow(print_img, cmap='bone')
        plt.imshow(mask_255[i][..., 0], cmap='Purples', alpha=0.5)
        plt.title('mask')
        plt.show()
'''
CT = normalize(CT)
mask = normalize(mask)
mask = np.rint(mask)
mask = mask.astype('int')

# TESTING NEW DATA ADJUSTING TO 255
CT = CT_255 #/ 255
mask = mask_255 / 255
mask = np.rint(mask)
mask = mask.astype('int')
'''

for i in range(79):
    plt.subplot(1, 2, 1)
    plt.imshow(CT[i, :, :, :], cmap = 'bone')
    plt.subplot(1, 2, 2)
    plt.imshow(CT[i, :, :, :], cmap= 'bone')
    plt.imshow(mask[i, :, :, :], alpha= 0.5, cmap='Purples')
    plt.show()

'''
# Plotting an example from the dataset
fig = plt.figure(figsize=(18, 15))

plt.subplot(1, 2, 1)
plt.imshow(CT[60][..., 0], cmap='bone')
plt.title('original CT')

plt.subplot(1, 2, 2)
#plt.imshow(CT[200][..., 0], cmap='bone')
plt.imshow(mask[60][..., 0], alpha=1, cmap="bone")
plt.title('original ovarian mask')
plt.show()

print(CT.shape)
print(mask.shape)

# DIVIDING IMAGE BATCHES
# Splitting the training data batches and the remain data
# x_train, x_rem, y_train, y_rem = split_(CT, mask, train_size)
# Splitting the remain data into validation data and test data
# x_valid, x_test, y_valid, y_test = split_(x_rem, y_rem, test_size)
x_train, x_remain, y_train, y_remain = split_(CT, mask, train_size)
x_test, x_valid, y_test, y_valid = split_(x_remain, y_remain, test_size)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)
print('x_valid: ', x_valid.shape)
print('y_valid: ', y_valid.shape)

# Saving all data in H5 file for convenience


with h5py.File('P:/PycharmProjects/pythonProject/storing/data.hdf5', 'w') as hf:
    x_train_file = hf.create_dataset('x_train', data=x_train,
                                     shape=x_train.shape, compression='gzip', chunks=True)
    y_train_file = hf.create_dataset('y_train', data=y_train,
                                     shape=y_train.shape, compression='gzip', chunks=True)
    x_test_file = hf.create_dataset('x_test', data=x_test,
                                    shape=x_test.shape, compression='gzip', chunks=True)
    y_test_file = hf.create_dataset('y_test', data=y_test,
                                    shape=y_test.shape, compression='gzip', chunks=True)
    x_valid_file = hf.create_dataset('x_valid', data=x_valid,
                                     shape=x_valid.shape, compression='gzip', chunks=True)
    y_valid_file = hf.create_dataset('y_valid', data=y_valid,
                                     shape=y_valid.shape, compression='gzip', chunks=True)

