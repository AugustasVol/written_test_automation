import numpy as np
import cv2

from base64image import Base64Image

def median_blur(img, kernel_size=3):
    return cv2.medianBlur(img,kernel_size)

def gaussian_blur(img, kernel_size = (3,3)):
    return cv2.GaussianBlur(img,kernel_size,0)
def odd_number(num):
    num = int(num)
    if (num % 2 == 0):
        num += 1
    return num

def resize(img, dims):
    if len(img.shape) == 3:
        img = img[0]
        img = np.array([cv2.resize(img,dims)])
    elif len(img.shape) == 4:
        img = img[0,0]
        img = np.array( [[cv2.resize(img,dims)]] )
    elif len(img.shape) == 2:
        img = cv2.resize(img,dims)
    else:
        raise

    return img
    
def rotate_bound(image, angle, border_white = False):

    if image.dtype == np.uint8 and border_white:
        borderValue = 255
    elif image.dtype == np.float64 and border_white:
        borderValue = 1.0
    elif image.dtype == np.bool and border_white:
        borderValue = 1
    else:
        borderValue = 0
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)

def shrink(img, max_size = 2000):
    '''resize img if one dimension is larger than max_size'''
    shape = img.shape
    index = np.argmax(shape)
    if shape[index] > max_size:
        ratio = max_size / shape[index]
        
        new_dims = ( int(shape[1] * ratio), int(shape[0] * ratio) ) # first 1 and second 0
        img = cv2.resize(img, new_dims)
    return img
def shrink_pil(img, max_size = 2000):

    dims = img.size
    max_dim = np.max(dims)
    if max_dim > max_size:
        ratio = max_size / max_dim
        dims = int(img.size[0] * ratio) , int(img.size[1] * ratio)
        img = img.resize(dims)
    return img


def imread(path, max_size = 2000):
    im = cv2.imread(path, 0) # read black and white picture, dtype=np.uint8
    if type(im) != type(None):
        im = shrink(im, max_size=max_size) # resize for faster computation
        return im
    else:
        raise
def imread_uri(uri, max_size = 2000):
    pil_image = Base64Image.from_uri(uri).get_pil_image()
    pil_image = shrink_pil(pil_image, max_size=max_size)
    pil_image = pil_image.convert(mode="L")

    np_image = np.array(pil_image)

    return np_image

def threshold(img, C = 3, kernel_size = 51):
    '''picture dtype=np.uint8'''
    if img.dtype == np.uint8:
        return cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,kernel_size,C)
    elif img.dtype == np.float64:
        img = img.astype(np.uint8) * 255
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,kernel_size,C)

 