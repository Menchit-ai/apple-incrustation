from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

os.chdir("./monodepth2")

import torch
from torchvision import transforms

import networks
import cmapy
import cv2 as cv

def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


class depth_model():
       
    def __init__(self, model_name="mono_640x192"):
        self.model_name = model_name
        download_model_if_doesnt_exist(self.model_name)
        encoder_path = os.path.join("models", self.model_name, "encoder.pth")
        depth_decoder_path = os.path.join("models", self.model_name, "depth.pth")

        self.imap = None
        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()
        os.chdir("..")
    
    
    def get_image(self) : return self.input_image
    def get_imap(self) : return self.imap
    
    def load_image(self,image_path):
        """
        Give a filename to be open using pillow, this will create the imap which can be shown by the function show_image()
        
        Argument : filename
        """
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
        step = int(_max/size)
        inter = np.arange(0,_max+(_max%step),step)
        res = []
        print("Starting extraction")
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