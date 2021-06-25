import argparse
import math
import os
import pickle
import random
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import script.library as lib
from script.depth_model import depth_model

parser = argparse.ArgumentParser()
parser.add_argument("background", help="the path where the background images are stored",type=str)
parser.add_argument("objects", help="the path where the objects that have to be incrusted are stored",type=str)
parser.add_argument("number", help="how many images you want to generate", type=int)
parser.add_argument("-o","--output", default="./output", help="where the generated images will be stored", type=str)
parser.add_argument("-i","--iteration", default=1, help="how many object we want at most, to incrust in one background", type=int)
# parser.add_argument("-v", "--verbose", default=0, help="0 for no output, 1 to print only the result, 2 for full output", type=int)
args = parser.parse_args()

workdir = os.getcwd()

if not os.path.isdir(args.background): raise Exception(args.background + " not found")
if not os.path.isdir(args.objects): raise Exception(args.objects + " not found")
if args.number<=0 : raise Exception("invalid number :",args.number)
if args.iteration<=0: raise Exception("invalid number :",args.iteration)

def main():
    # recover all the backgrounds' and objects' filename
    backgrounds = [os.path.join(args.background,f) for f in os.listdir(args.background)]
    objects = [os.path.join(args.objects,f) for f in os.listdir(args.objects)]

    if len(backgrounds)==0 : raise Exception(args.background,"is empty")
    if len(objects)==0 : raise Exception(args.objects,"is empty")

    if not os.path.isdir(".light") : os.mkdir(".light")
    if not os.path.isdir(".cropped_objects") : os.mkdir(".cropped_objects")

    # compute the starting and ending point of the light vector for each image and store them in a .light folder
    for background in backgrounds:
        light_vector = lib.light_vectorization(background)
        filename = os.path.basename(background).split('.')[0] + '.bin'
        with open('.light/'+filename,'wb') as f : pickle.dump(light_vector,f)
    
    # cropping all the object (using black pixels) and store the new files in binaries in the .cropped_objects
    for object in objects:
        cropping_object = lib.cropping_object(object)
        filename = os.path.basename(object).split('.')[0] + '.bin'
        with open('.cropped_objects/'+filename,'wb') as f : pickle.dump(cropping_object,f)


if __name__ == "__main__":
    main()
