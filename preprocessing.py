'''
Preprocessing file when dealing with input data type .nii
- Adjust the aspect ratio,
- Saving image matrix
'''
import random

from library import *


def preProcessing(filePath, outputPath, file_name, list_, axis_flag, datatype, first_index, amount_of_slices):
    '''

    scan = nib.load(filePath)
    scanArray = scan.get_fdata()
    scanArrayShape = scanArray.shape
    print('Scan data array has the shape: ', scanArrayShape)

    #THIS IS FOR THE FULL 3D IMAGE
    for i in range(scanArrayShape[0]):
        outputArray = cv2.resize(scanArray[i, :, :], (scanArrayShape[2], scanArrayShape[1]))
        cv2.imwrite(outputPath + '/Dim0_Slice' + str(i) + '.png', outputArray)

    for i in range(scanArrayShape[1]):
        outputArray = cv2.resize(scanArray[:, i, :], (scanArrayShape[2], scanArrayShape[0]))
        cv2.imwrite(outputPath + '/Dim1_Slice' + str(i) + '.png', outputArray)

    # only 3rd dimension (allowing this only for faster computing time)
    for i in range(scanArrayShape[2]):
        outputArray = cv2.resize(scanArray[:, :, i], (scanArrayShape[1], scanArrayShape[0]))
        #outputArray = cv2.rotate(outputArray, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(outputPath + '/' + str(file_name) + '_Dim2_Slice' + str(i) + '.png', outputArray)

    '''
    scan = nib.load(filePath)
    scanArray = scan.get_fdata()
    # scanArray = np.rot90(np.array(scanArray))
    scanArrayShape = scanArray.shape
    print(scanArrayShape)
    imgsize = first_index
    if axis_flag == 'x':
        for i in range(amount_of_slices):
            img = cv2.resize(scanArray[imgsize, :, :],
                             dsize=(512, 512),
                             interpolation=cv2.INTER_AREA).astype(datatype)
            imgsize = imgsize + 1
            list_.append(img[..., np.newaxis])

    if axis_flag == 'y':
        for i in range(amount_of_slices):
            img = cv2.resize(scanArray[:, imgsize, :],
                             dsize=(512, 512),
                             interpolation=cv2.INTER_AREA).astype(datatype)
            imgsize = imgsize + 1
            list_.append(img[..., np.newaxis])

    if axis_flag == 'z':
        for i in range(amount_of_slices):
            img = cv2.resize(scanArray[..., imgsize],
                             dsize=(512, 512),
                             interpolation=cv2.INTER_AREA).astype(datatype)
            imgsize = imgsize + 1
            list_.append(img[..., np.newaxis])
    return list_


'''
def resize_(filepath, outputpath, size, path):
    img = cv2.imread(filepath)
    resized_img = cv2.resize(img.astype('float'), (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outputpath + str(path) + '_resize.png', resized_img)
'''


def split_(batch_x, batch_y, ratio):
    batch_x1, batch_x2, batch_y1, batch_y2 = train_test_split(batch_x, batch_y, train_size=ratio)
    return batch_x1, batch_x2, batch_y1, batch_y2


def normalize(img):
    # min = input.min(axis = (1, 2, 3), keepdims = True)
    # max = input.max(axis = (1, 2, 3), keepdims = True)
    norm_input = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm_input


def load_img_from_folder(folder):
    images = np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images = np.append(images, img)
    return images


# DATA AUGMENTATION
def rotate(img, angle):
    #angle = int(random.uniform(-angle, angle))
    #h, w = 512, 512
    #M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    #img = cv2.warpAffine(img, M, (w, h))
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def flip_vertically(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


def add_noise(img):
    # adding Gaussian noise to the images
    x, y = 512, 512
    mean = 0
    var = 0.01
    sigma = np.sqrt(var)
    noise = np.random.normal(loc=mean,
                             scale=sigma,
                             size=(x, y))
    noisy_img = img + noise
    return noisy_img


def adjust(img):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img


def inverse_log_transform(img):
    L = 255
    c = L / (np.log(1 + L))
    y = np.exp(img ** 1 / c) - 1
    return y

def jere(img):
    img = 255 - img
    img = np.log(img)
    return img