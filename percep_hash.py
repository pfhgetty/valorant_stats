import cv2
import numpy as np

def hash_function(im):
    return np.unpackbits(cv2.img_hash.blockMeanHash(im))

def pHash(im, debug=False):
    h, w, _ = im.shape
    x, y = w // 3, h // 3
    im_portion = im[y: y*2, x : 2 * x, :]
    im_portion = cv2.cvtColor(cv2.GaussianBlur(im_portion, (0,0), 1), cv2.COLOR_BGR2GRAY)
    hashed = hash_function(im_portion)
    if debug:
        return hashed, im_portion
    return hashed
