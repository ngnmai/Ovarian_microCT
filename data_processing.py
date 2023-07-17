'''
- Processing all the data files into usable file types
- Divided data into batches
- Augmenting data if needed
'''
# Importing necessary libraries
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
train_size = 0.8
test_size = 0.5

# size of the images that we want to resize to
img_size = 512

images_path = ['01032022', '01092022_AreaC', '01092022_AreaD', '02092022', '04022021', '05022021']
labels_path = ['01032022_seg', '01092022_areaC_seg', '01092022_areaD_seg', '02092022_seg', '04022021_seg', '05022021_seg']
start_index = [29, 99, 499, 749, 319, 419]
amount_toTrim = [20, 10, 15, 15, 10, 15]
axis_ = ['x', 'y', 'x', 'z', 'z', 'y']

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
        infection.append(fullpath)
print("y_train ", len(infection))
'''



CT = np.array(CT)
mask = np.array(mask)

# Plotting an example from the dataset
fig = plt.figure(figsize=(18, 15))

plt.subplot(1, 2, 1)
plt.imshow(CT[5][..., 0], cmap='bone')
plt.title('original CT')

plt.subplot(1, 2, 2)
plt.imshow(CT[5][..., 0], cmap='bone')
plt.imshow(mask[5][..., 0], alpha=0.5, cmap="nipy_spectral")
plt.title('original ovarian mask')
plt.show()

print(CT.shape)
print(mask.shape)

# NORMALIZING DATA
# no need to normalize the mask
CT = normalize(CT)
mask = normalize(mask)
mask = np.rint(mask)
mask = mask.astype('int')

# DIVIDING IMAGE BATCHES
# Splitting the training data batches and the remain data
# x_train, x_rem, y_train, y_rem = split_(CT, mask, train_size)
# Splitting the remain data into validation data and test data
# x_valid, x_test, y_valid, y_test = split_(x_rem, y_rem, test_size)
x_train, x_test, y_train, y_test = split_(CT, mask, train_size)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)

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

