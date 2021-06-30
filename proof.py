import os
import shutil

import cv2 as cv
from tqdm.auto import tqdm


def main(output,proof):
    # store all the images that were created by the algorithm
    files = [f.split('.')[0] for f in os.listdir(output)][::2]
    jpg = [f+".jpg" for f in files]
    txt = [f+".txt" for f in files]

    # create the folder that will store the proof images
    if os.path.isdir(proof): shutil.rmtree(proof)
    os.mkdir(proof)

    for i in tqdm(range(len(files))):
        # select the image file and the file with the coordinates
        j = os.path.join(output,jpg[i])
        t = os.path.join(output,txt[i])
        lines = open(t,'r').read().splitlines()
        img = cv.imread(j)
        
        # draw rectangle using the coordinates found in the .txt file
        for line in lines:
            _,xs,xe,ys,ye = list(map(int,line.split(' ')))
            img = cv.rectangle(img,(ys,xs),(ye,xe),(0,255,0),2)
        
        cv.imwrite(proof+"/"+files[i]+".jpg",img)
