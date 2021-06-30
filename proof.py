import os
import shutil

import cv2 as cv
from tqdm.auto import tqdm


def main(output,proof):
    files = [f.split('.')[0] for f in os.listdir(output)]
    jpg = [f+".jpg" for f in files]
    txt = [f+".txt" for f in files]

    if os.path.isdir(proof): shutil.rmtree(proof)
    os.mkdir(proof)

    for i in tqdm(range(len(files))):
        j = os.path.join(output,jpg[i])
        t = os.path.join(output,txt[i])
        lines = open(t,'r').read().splitlines()
        img = cv.imread(j)
        
        for line in lines:
            _,xs,xe,ys,ye = list(map(int,line.split(' ')))
            img = cv.rectangle(img,(ys,xs),(ye,xe),(0,255,0),2)
        
        cv.imwrite(proof+"/"+files[i]+".jpg",img)
