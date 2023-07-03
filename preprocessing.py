'''
Preprocessing file when dealing with input data type .nii
- Adjust the aspect ratio,
- Saving image matrix
'''
from library import *

# filepath = 'P:/StudentDocuments/Documents/TESTING DATA/ct_scans/coronacases_org_001.nii'

def preProcessing(filePath, outputPath, file_name, list_, flag, datatype):
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
    scanArray = np.rot90(np.array(scanArray))
    scanArrayShape = scanArray.shape
    print(scanArrayShape)
    for imgsize in range(scanArrayShape[2]):
        img = cv2.resize(scanArray[..., imgsize],
                         dsize=(512, 512),
                         interpolation=cv2.INTER_AREA).astype(datatype)
        list_.append(img[..., np.newaxis])
        #if flag:
            #cv2.imwrite(outputPath + '/' + str(file_name) + '_Dim2_Slice' + str(imgsize) + '.png', img)
    return list_

def resize_(filepath, outputpath, size, path):
    img = cv2.imread(filepath)
    resized_img = cv2.resize(img.astype('float'), (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outputpath + str(path) + '_resize.png', resized_img)


def split_(batch_x, batch_y, ratio):
    batch_x1, batch_x2, batch_y1, batch_y2 = train_test_split(batch_x, batch_y, train_size=ratio)
    return batch_x1, batch_x2, batch_y1, batch_y2

def normalize(input):
    min = input.min(axis = (1, 2, 3), keepdims = True)
    max = input.max(axis = (1, 2, 3), keepdims = True)
    norm_input = (input - min) / (max - min)
    return norm_input

def load_img_from_folder(folder):
    images = np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images = np.append(images, img)
    return images
