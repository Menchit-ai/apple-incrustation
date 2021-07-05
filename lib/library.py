
import random
import warnings

import cv2 as cv
import numpy as np
from PIL import Image
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

def cropping_object(file):
    """
    Crop an object image by taking the first non black pixel as first coordinate for each axis
    
    Argument : image
    Return : cropped image object
    """

    img = cv.imread(file)
    positions = np.nonzero(img)

    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()

    return img[top:bottom,left:right,:]

def show(matrix):
    """
    Convert bgr to rgb matrix

    Argument : 3D matrix
    Return : 3D matrix
    """

    if len(matrix.shape)==3 : matrix = matrix[:,:,::-1]
    return Image.fromarray(matrix)

def add_noise(img,iterations=1):
    """
    Adding random noise in image by turning some black

    Arguments : image (matrix), (optionnal) number of time we want to add noise
    Return : noised matrix
    """

    # Getting the dimensions of the image
    row , col = img.shape

    for i in range(iterations):
      number_of_pixels = random.randint(300 , 10000)
      for i in range(number_of_pixels):
          
          # Pick a random y coordinate
          y_coord=random.randint(0, row - 1)
            
          # Pick a random x coordinate
          x_coord=random.randint(0, col - 1)
            
          # Color that pixel to black
          img[y_coord][x_coord] = 0
          
    return img

def extract(im, size,coord=None):
    """
    Extract an area in an image

    Arguments : image (matrix), size of the area to be extracted (in pixels), (optionnal) coordinates of the starting point (if there is one)
    """

    if not len(size) == 2: raise Exception(size, "must be of len=2")
    _shape = im.shape
    if _shape[0] < size[0] or _shape[1] < size[1]:
        raise Exception("Square to big for image : ", _shape[:-1], size)
    
    im = im.copy()
    x_max,y_max = _shape[0]-size[0], _shape[1]-size[1]
    
    if coord is None : x_start,y_start = random.randint(0,x_max+1),random.randint(0,y_max+1)
    else : x_start,y_start = coord
    x_end,y_end = x_start+size[0],y_start+size[1]
    
    return im[x_start:x_end,y_start:y_end,:],(x_start,x_end),(y_start,y_end)

def light_vectorization(image):
    """
    Create a vector that will be used to estimate where the light hit an object

    Argument : image (matrix)
    Return : starting point and ending point of the light vector
    """

    image = cv.imread(image,cv.IMREAD_GRAYSCALE)
    thresh = cv.threshold(image, 160, 255, cv.THRESH_BINARY)[1]
    thresh = add_noise(thresh, iterations=5)
    thresh = thresh - cv.erode(thresh, None, iterations=1)
    
    output = cv.connectedComponentsWithStats(thresh, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    centroids = centroids.astype(int)
    
    x = centroids[:,0].reshape(-1,1)
    y = centroids[:,1]
    gbr = XGBRegressor(random_state=42)
    gbr.fit(x,y)
    start = (0,int(gbr.predict(np.asarray([[0]]))))
    end = (image.shape[1],int(gbr.predict(np.asarray([[image.shape[1]]]))))
    return start,end


def compute_impact(image,coord,vect):
    """
    Compute where the light is hitting a box using light vector

    Arguments : image (matrix), coordinates of the box to study, vector that represents the light
    Return : coordinates of the area where the light is hitting
    """

    x,y = coord
    start, end = vect
    if start[1] < end[1] : light, dark = np.asarray(start), np.asarray(end)
    else : light, dark = np.asarray(end), np.asarray(start)

    if light[0]==0:
        vect = (dark-light)/(image.shape[0])*x[0]
        locy = vect[1]/image.shape[1] * (y[1]-y[0])
        impact = np.asarray([x[0],locy]).astype(np.float32)

    elif light[1]==0:
        pos = light[0]/image.shape[0]
        impact = np.asarray([x[0],(y[1]-y[0])*pos]).astype(np.float32)

    else:
        side = np.asarray([[x[1],y[0]],[x[1],y[1]]])
        vect = (dark-light)/(image.shape[0])*(image.shape[0]-x[1])
        locy = vect[1]/image.shape[1] * (y[1]-y[0])
        impact = np.asarray([x[0],locy]).astype(np.float32)

    return (impact - [x[0],y[0]]).astype(np.uint8)

def report_impact(image,coord,vect):
    """
    Compute where the light is hitting a box using light vector

    Arguments : image (matrix), coordinates of the box to study, vector that represents the light
    Return : coordinates of the area where the light is hitting
    """

    x,y = coord
    start, end = vect
    if start[1] < end[1] : light, dark = np.asarray(start), np.asarray(end)
    else : light, dark = np.asarray(end), np.asarray(start)

    if light[0]==0:
        position = start[1]/image.shape[1]
        locy = (y[1]-y[0])*position
        impact = np.asarray([0,locy])

    elif light[1]==0:
        side = np.asarray([[x[0],y[0]],[x[1],y[0]]])
        pos = light[0]/image.shape[0]
        impact = np.asarray([x[0],(y[1]-y[0])*pos]).astype(np.float32)

    else:
        position = start[1]/image.shape[1]
        locy = y[0] + (y[1]-y[0])*position
        impact = np.asarray([image.shape[0],locy-y[0]]).astype(np.float32)

    return impact.astype(np.uint8)


def create_light_v2(image,position):
    """
    Create light mask by drawing circles, one is getting whither and the opposite is getting darker

    Arguments : image (matrix), position of the light source (withe circle's center)
    Return : light mask
    """

    import math
    import cv2 as cv
    
    x,y,_ = image.shape
    image = np.ones((x,y))*0.5
    # Center coordinates
    light = np.asarray(position)
    dark = np.asarray([x,y]) - position

    # Radius of circle
    p = 13
    d = 16
    radius = max(int(math.sqrt( (light[0]-dark[0])**2+(light[1]-dark[1])**2)/2),1)
    light_radius = int(math.sqrt( (light[0]-dark[0])**2+(light[1]-dark[1])**2)/d )*p
    dark_radius = int(math.sqrt( (light[0]-dark[0])**2+(light[1]-dark[1])**2)/d )*(d-p)
    thickness = -1
    
    ran = np.arange(0,0.5,(0.5/radius))
    
    for i in range(radius):
        new_brightness = min(0.5+ran[i],1)
        new_darkness = max(0.5-ran[i],0)
        new_light_radius = max(light_radius-i,0)
        new_dark_radius = max(dark_radius-i,0)       
        image = cv.circle(image, tuple(dark), new_dark_radius, new_darkness, thickness)
        image = cv.circle(image, tuple(light), new_light_radius, new_brightness, thickness)

    return image

def create_light_v3(image,position):
    """
    Create light mask by drawing circles, one is getting whither and the opposite is getting darker

    Arguments : image (matrix), position of the light source (withe circle's center)
    Return : light mask
    """

    import math
    import cv2 as cv
    
    x,y,_ = image.shape
    image = np.ones((x,y))*0
    # Center coordinates
    light = np.asarray(position)
    dark = np.asarray([x,y]) - position

    # Radius of circle
    p = 11
    d = 16
    radius = int(math.sqrt( (light[0]-dark[0])**2+(light[1]-dark[1])**2)/2)
    light_radius = int(math.sqrt( (light[0]-dark[0])**2+(light[1]-dark[1])**2)/d )*p
    dark_radius = int(math.sqrt( (light[0]-dark[0])**2+(light[1]-dark[1])**2) )
    dark_radius = int(math.sqrt( (x-0)**2+(y-0)**2))
    thickness = -1
    
    ran = np.arange(0,0.5,(0.5/radius))
    
    for i in range(radius):
        new_brightness = min(0.5+ran[i],1)
        new_darkness = min(0+ran[i],255)
        new_light_radius = max(light_radius-i,0)
        new_dark_radius = max(dark_radius-i,0)   
        if new_dark_radius <= 0 : new_dark_radius=0

        image = cv.circle(image, tuple(light), new_light_radius, new_brightness, thickness)
        image = cv.circle(image, tuple(light), new_dark_radius, new_darkness, thickness)
        
    for i in range(radius):
        new_brightness = min(0.5+ran[i],1)
        new_darkness = min(0+ran[i],255)
        new_light_radius = max(light_radius-i,0)
        new_dark_radius = max(dark_radius-i,0)   

        image = cv.circle(image, tuple(light), new_light_radius, new_brightness, thickness)

    return image

def add_parallel_light(image, mask):
    """
    Apply the light mask to an image

    Arguments : image (matrix), light mask which has to be apply
    Return : image where the mask is applied
    """
    
    frame = image.copy()
    height, width, _ = frame.shape
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    adjust = 255- np.max(hsv[:,:,2])
    hsv[:,:,2] = hsv[:,:,2]+adjust
    hsv[:, :, 2] = hsv[:, :, 2]*mask
    hsv[hsv>255] = 255
    hsv[hsv<0] = 0
   
    frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    frame = np.asarray(frame, dtype=np.uint8)
    return frame.astype(np.uint8)

def channel_operand(im,value,operand):
    """
    Computing operation along all axis of a 3D matrix

    Arguments : matrix, value to use in calculation, which operand use
    Return : 3D matrix where the calculation has been applied
    """

    operand = operand.lower()
    
    b,g,r = im[:,:,0],im[:,:,1],im[:,:,2]
    
    if operand=="add": b,g,r = b+value,g+value,r+value
    elif operand=="sub": b,g,r = b-value,g-value,r-value
    elif operand=="mult": b,g,r = b*value,g*value,r*value
    elif operand=="div": b,g,r = b/value,g/value,r/value
    else: raise Exception("Choose between add, sub, mult or div.")
    
    return np.dstack((b,g,r))

def brightness_matrix(image):
    """
    Return a gray image from colour image

    Argument : colour image (3D matrix)
    Return : gray image (2D matrix)
    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray/255


def adjust_brightness(im1,im2,kernel=None):
    """
    Compute the difference of light between two image on a local area

    Arguments : first image (3D matrix), second image (3D matrix), (optionnal) size of the local area to be checked
    Return : matrix of differences between the two images
    """

    brightness1 = brightness_matrix(im1)
    brightness2 = brightness_matrix(im2)
    x_max,y_max,_ = im1.shape
    if kernel is None: return brightness2 - brightness1
    
    x_step,y_step = kernel
    light = np.zeros((x_max,y_max))
    
    for y in range(y_max)[::y_step]:
        for x in range(x_max)[::x_step]:
            pixels = brightness1[x:x+x_step,y:y+y_step]
            light[x:x+x_step,y:y+y_step] = np.mean(brightness2[x:x+x_step,y:y+y_step]) - np.mean(pixels)
                

    return light

def combine(im1, im2, light, kernel=(10,10)):
    """
    Combining two images by incrusting im2 in im1, im2's brightness is adjusted depending on im1's mean brightness

    Argument : background image, object to incrust, lightened object to be incrusted, kernel used during erosion
    Return : image where the object is incrusted
    """

    shape = im1.shape
    
    gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

    kernel = np.ones(kernel, np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=1)

    mask = cv.resize(thresh, (im1.shape[1], im1.shape[0])).astype(bool)
    
    im2 = cv.resize(im2, (im1.shape[1], im1.shape[0]))
    light = cv.resize(light, (im1.shape[1], im1.shape[0]))

    brightness = brightness_matrix(im1).max()
    dimed_im2 = np.dstack(
        (im2[:, :, 0] * brightness, im2[:, :, 1] * brightness,
         im2[:, :, 2] * brightness)).astype(np.uint8)

    final_image = np.zeros(shape)
    for k in range(final_image.shape[2]):
        for i in range(final_image.shape[0]):
            for j in range(final_image.shape[1]):
                if not mask[i, j]: final_image[i, j, k] = im1[i, j, k]
                else: final_image[i, j, k] = light[i, j, k]

    final_image = final_image.astype(np.uint8)

    return final_image

def combinev2(im1, im2,kernel=(10,10)):
    """
    Combining two images by incrusting im2 in im1, im2's brightness is adjusted depending on im1's mean brightness in a local area

    Argument : background image, object to incrust, lightened object to be incrusted, kernel used during erosion
    Return : image where the object is incrusted
    """

    shape = im1.shape
    
    gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones(kernel, np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=1)
    mask = cv.resize(thresh, (im1.shape[1], im1.shape[0])).astype(bool)
    
    im2 = cv.resize(im2, (im1.shape[1], im1.shape[0]))

    brightness = adjust_brightness(im1,im2,(10,10))
    contrast = adjust_contrast(im1,im2,(10,10))
    im2 = channel_operand(im2,contrast,"mult")
    im2 = channel_operand(im2,brightness,"add").astype(np.uint8)
    im2 = cv.GaussianBlur(im2,(11,11),0)
    
    final_image = np.zeros(shape)
    for k in range(final_image.shape[2]):
        for i in range(final_image.shape[0]):
            for j in range(final_image.shape[1]):
                if not mask[i, j]: final_image[i, j, k] = im1[i, j, k]
                else: final_image[i, j, k] = im2[i, j, k]

    final_image = final_image.astype(np.uint8)

    return final_image

def combinev3(background, object, light_object, kernel=(10,10)):
    """
    Combining two images by incrusting im2 in im1

    Argument : background image, object to incrust, lightened object to be incrusted, kernel used during erosion
    Return : image where the object is incrusted
    """

    shape = background.shape
    
    gray = cv.cvtColor(object, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones(kernel, np.uint8)
    # thresh = cv.erode(thresh, kernel, iterations=1).astype(np.uint8)
    mask = cv.resize(thresh, (background.shape[1], background.shape[0])).astype(bool)
    
    object = cv.resize(object, (background.shape[0], background.shape[0]))
   
    final_image = np.zeros(shape)
    for k in range(final_image.shape[2]):
        for i in range(final_image.shape[0]):
            for j in range(final_image.shape[1]):
                if not mask[i, j]: final_image[i, j, k] = background[i, j, k]
                else: final_image[i, j, k] = light_object[i, j, k]

    final_image = final_image.astype(np.uint8)

    return final_image

def reconstruct(images):
    """
    Combine several images together by taking the first image and adding all the non black pixels of the next one and continuing for all the images

    Argument : list of images (3D matrix)
    Return : image which is the combination of all the images in the list (3D matrix)
    """

    new_image = np.zeros(images[0].shape)   
    for im in images:
        new_image = paste_non_black(new_image,im)
    return new_image

def paste_non_black(im1,im2):
    """
    Taking all the non black pixels of the second image and replacing the same pixels of the first image by them

    Arguments : first image which will be the background, second image which will be pasted
    Return : image (3D matrix)
    """

    base = im1.copy()
    _2add = im2.copy()

    non_black_indices = np.nonzero(cv.cvtColor(_2add,cv.COLOR_BGR2GRAY))
    base[non_black_indices] = _2add[non_black_indices]

    return base.astype(np.uint8)
