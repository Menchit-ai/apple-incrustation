import argparse
import math
import os
import random
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import script.apple_incrustation as incrust
import script.depth_model as depth
import script.light_application as lightapp
import script.light_estimation as lightest

parser = argparse.ArgumentParser()
parser.add_argument("background", help="the path where the background images are stored",type=str)
parser.add_argument("objects", help="the path where the objects that have to be incrusted are stored",type=int)
parser.add_argument("number", help="how many images you want to generate", type=int)
parser.add_argument_group("-o","--output", default="./output", help="where the generated images will be stored", type=str)
parser.add_argument_group("-i","--iteration", default=1, help="how many object we want at most, to incrust in one background", type=int)
parser.add_argument("-v", "--verbose", default=0, help="0 for no output, 1 to print only the result, 2 for full output", type=int)
args = parser.parse_args()

if not os.path.isdir(args.background): raise Exception(args.background, "not found")
if not os.path.isdir(args.objects): raise Exception(args.objects, "not found")
if args.number<=0 or args.iteration<=0: raise Exception("invalid number :",args.number)

def main():
    pass



if __name__ == "__main__":
    main()
