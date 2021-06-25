import argparse
import math
import os
import pickle
import random
import shutil
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from torch.nn.modules import module
from tqdm.auto import tqdm

import lib.library as lib
from lib.depth_model import depth_model

parser = argparse.ArgumentParser()
parser.add_argument("background", help="the path where the background images are stored",type=str)
parser.add_argument("objects", help="the path where the objects that have to be incrusted are stored",type=str)
parser.add_argument("number", help="how many images you want to generate", type=int)
parser.add_argument("-o","--output", default="./output", help="where the generated images will be stored", type=str)
parser.add_argument("-i","--iteration", default=1, help="how many object we want at most, to incrust in one background", type=int)
parser.add_argument("-d","--depths", default=4, help="Set how many depths we want to use in our background image", type=int)
args = parser.parse_args()

workdir = os.getcwd()

if not os.path.isdir(args.background): raise Exception(args.background + " not found")
if not os.path.isdir(args.objects): raise Exception(args.objects + " not found")
if args.number<=0 : raise Exception("invalid number :",args.number)
if args.iteration<=0: raise Exception("invalid number :",args.iteration)

model = depth_model()
console = Console()

def main():
    # recover all the backgrounds' and objects' filename
    backgrounds = [os.path.join(args.background,f) for f in os.listdir(args.background)]
    objects = [os.path.join(args.objects,f) for f in os.listdir(args.objects)]

    if len(backgrounds)==0 : raise Exception(args.background,"is empty")
    if len(objects)==0 : raise Exception(args.objects,"is empty")

    if not os.path.isdir(".light") : os.mkdir(".light")
    if not os.path.isdir(".cropped_objects") : os.mkdir(".cropped_objects")
    if not os.path.isdir(".depths") : os.mkdir(".depths")
    if not os.path.isdir(".imaps") : os.mkdir(".imaps")
    if os.path.isdir(args.output) : shutil.rmtree(args.output)
    os.mkdir(args.output)
    output_folder = args.output

    # compute the starting and ending point of the light vector for each image and store them in a .light folder
    for background in backgrounds:
        light_vector = lib.light_vectorization(background)
        model.load_image(background)
        depths = model.extract_depth(args.depths)
        imap = model.get_imap()
        filename = os.path.basename(background).split('.')[0] + '.bin'
        with open('.light/'+filename,'wb') as f : pickle.dump(light_vector,f)
        with open('.depths/'+filename,'wb') as f : pickle.dump(depths,f)
        with open('.imaps/'+filename,'wb') as f : pickle.dump(imap,f)
    
    # cropping all the object (using black pixels) and store the new files in binaries in the .cropped_objects
    for object in objects:
        cropping_object = lib.cropping_object(object)
        filename = os.path.basename(object)
        cv.imwrite(os.path.join('.cropped_objects',filename),cropping_object)

    # looping on how many images we want to generate
    
    console.print("\n"+
        "Begining extraction"+
        "\n", style = "bold red")

    for i in range(args.number):
        background = random.choice(backgrounds)
        back_name = os.path.basename(background).split('.')[0]
        img_filename = os.path.join(output_folder,"_".join([back_name,str(i),".jpg"]))
        area_filename = os.path.join(output_folder,"_".join([back_name,str(i),".txt"]))
        console.print(img_filename, style = "bold blue")
        open(img_filename,'w').close()
        open(area_filename,'w').close()
        cv.imwrite(img_filename,cv.imread(background))

        # incrusting a random number of objects between 1 and the iteration argument of the script
        for j in tqdm(range(random.randint(1,args.iteration))):
            object = random.choice(objects)
            incrustation,area = incrust(background, object)
            with open(area_filename,'a') as f : f.write(area+"\n")
            cv.imwrite(img_filename,incrustation)


def incrust(background,object):
    bin_background = os.path.basename(background).split('.')[0]+'.bin'
    bin_object = os.path.basename(object).split('.')[0]+'.bin'

    with open(os.path.join(".depths",bin_background),'rb') as f : depths = pickle.load(f)
    with open(os.path.join(".imaps",bin_background),'rb') as f : imap = pickle.load(f)
    with open(os.path.join(".light",bin_background),'rb') as f : light_vector = pickle.load(f)
    back_image = cv.imread(background)
    object_image = cv.imread(os.path.join(".cropped_objects",os.path.basename(object)))

    back_shape = back_image.shape
    object_shape = object_image.shape

    # define which depth will be used to put the object on
    where = random.randint(0,len(depths)-1)
    gray = cv.cvtColor(depths[where],cv.COLOR_BGR2GRAY)
    points = gray.nonzero()
    which_point = random.randint(0,len(points[0])-1)
    # this point will be used to determine the diminution apply to our object
    point = (points[0][which_point],points[1][which_point])
    # scale = imap[point]/255
    scale = 1

    # we now have to apply our scale by making our object the greatest possible and then to reduce it with our scale

    coeff = min(back_shape)/object_shape[back_shape.index(min(back_shape))]
    height,width = object_shape[:-1]
    dim = (int(width*coeff*scale), int(height*coeff*scale))
    object_image = cv.resize(object_image,dim)
    object_shape = object_image.shape

    # now that we have our rescaled image we have to select an area where we will put our object
    # first we want to be sure that our ibject will fully fit in our image
    new_x, new_y = point
    if point[0] + object_shape[0] > back_shape[0] : new_x -= object_shape[0]
    if point[1] + object_shape[1] > back_shape[1] : new_y -= object_shape[1]
    if new_x <0 or new_y < 0 : Exception(new_x,new_y,point)
    starter = (new_x, new_y)
    area,x,y = lib.extract(depths[where],object_shape[:-1],starter)

    # now we know where we are going to incrust our object, we will first apply the lightning of our background image on our object
    lightened_object = light_object(back_image,light_vector,object_image,(x,y))
    area = lib.combinev3(area, object_image, lightened_object)
    back_image[x[0]:x[1],y[0]:y[1],:] = area
    return back_image," ".join([str(0),str(x[0]),str(x[1]),str(y[0]),str(y[1])])


def light_object(back_image,light_vector,object_image,coord):
    impact = lib.report_impact(back_image,coord,light_vector)
    light = lib.create_light_v2(object_image,impact)
    enlighted = lib.add_parallel_light(object_image,light)
    return enlighted

if __name__ == "__main__":
    console.print("\n"+
        "Starting creation using "+str(args.number)+" backgrounds."+
        "\n", style = "bold red")
    main()
