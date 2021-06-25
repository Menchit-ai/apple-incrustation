from __future__ import absolute_import, division, print_function

import hashlib
import os
import urllib
import zipfile

import cmapy
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
from tqdm.auto import tqdm

from .networks import ResnetEncoder, DepthDecoder
from .utils import download_model_if_doesnt_exist

class depth_model():
       
    def __init__(self, model_name="mono_640x192"):
        self.model_name = model_name
        download_model_if_doesnt_exist(self.model_name)
        encoder_path = os.path.join("models", self.model_name, "encoder.pth")
        depth_decoder_path = os.path.join("models", self.model_name, "depth.pth")

        self.imap = None
        # LOADING PRETRAINED MODEL
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()
    
    
    def get_image(self) : return self.input_image
    def get_imap(self) : return self.imap
    
    def load_image(self,image_path):
        """
        Give a filename to be open using pillow, this will create the imap which can be shown by the function show_image()
        
        Argument : filename
        """
        self.image_name = image_path
        self.input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = self.input_image.size

        feed_height = self.loaded_dict_enc['height']
        feed_width = self.loaded_dict_enc['width']
        input_image_resized = self.input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        
        with torch.no_grad():
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        
        disp_resized = torch.nn.functional.interpolate(disp,
            (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        self.imap = (disp_resized_np*255).astype(np.uint8)
        self.input_image = np.asarray(self.input_image).astype(np.uint8)
        
    def show_image(self):
        """
        Show the image currently loaded in the model
        """
        if self.input_image is None : raise Exception("No image loaded")
        plt.imshow(self.input_image)
        
    def show_imap(self):
        """
        Show the imap of the currently loaded image
        """
        if self.input_image is None : raise Exception("No image loaded")
        plt.imshow(self.imap)
        
        
    def extract_depth(self,size):
        """
        Return a 3D matrix which contains each plan in the image
        
        Argument : size (nb of plans that we want to differentiate)
        Return : matrix of shape (size,image.shape)
        """
        
        def threshold(image,down,up):
            assert len(image.shape)==2
            a,b = image.shape
            locmap = image.copy()
            
            for i in range(a):
                for j in range(b):
                    if not(down<=image[i][j] and image[i][j]<up): locmap[i][j]=0
                        
            return locmap
        
        if self.imap is None : raise Exception("No image loaded")
        
        _max = np.max(self.imap)
        step = int(_max/size+1)
        inter = np.arange(0,_max+step,step)
        res = []
        
        print("Creation of depth in "+self.image_name)
        for k in tqdm(range(len(inter)-1)):
            thresh = threshold(self.imap,inter[k],inter[k+1])
            res.append(thresh)
            
        depths = np.asarray(res).astype(bool)
        
        depth_in_image = []
        for depth in depths:
            b = self.input_image[:,:,0].copy()
            g = self.input_image[:,:,1].copy()
            r = self.input_image[:,:,2].copy()
            b = b*depth
            g = g*depth
            r = r*depth
            image = np.dstack((b,g,r))
            depth_in_image.append(image)
        
        return np.asarray(depth_in_image).astype(np.uint8)
