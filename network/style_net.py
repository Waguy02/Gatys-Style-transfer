import logging
import os
import shutil

import numpy as np
import pytorch_lightning
import torch
import torchvision.models
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torchvision.utils import save_image

from constants import ROOT_DIR
from network.super_resolution import ISRModel

device="cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
INPUT_SIZE=224
class StyleTransferNetwork(pytorch_lightning.LightningModule):
    def __init__(self,num_iter=1e4,lr=1e-2):
        super().__init__()
        self.layers_style={4, 7, 12, 21, 30}
        self.num_iter=int(num_iter)
        self.lr=lr
        self.setup_network()
        self.transform = transforms.Compose([transforms.Resize((INPUT_SIZE,INPUT_SIZE)), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])
                                             ])
        self.upsampler=ISRModel()

    def setup_network(self):
        """
        Setup the network,freeze all layers and replace MaxPooling with AvgPooling
            """
        self.vgg = torchvision.models.vgg19(pretrained=True)
        for params in self.vgg.parameters():  ##The model is frozen
            params.requires_grad = False
        for name, child in self.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)
        self.to(device).eval()
        logging.info("Device: {}".format(device))

    def forward(self, x):
        aux_features=[]
        for name, module in self.vgg.features.named_children():
            x=module(x)
            if int(name) in self.layers_style:
                aux_features.append(x)
        return x,aux_features


    def loss_content(self,x,y):
        l=0
        for i in range(len(x)):
            l+=torch.sum((x[i]-y[i])**2)
        return l/len(x)
    def loss_style(self,aux_features_x,aux_features_y):
        """
        Compute a style loss from gram matrix

        """
        def loss_style_single(aux_features_x,aux_features_y):
            (N,C,H,W)=aux_features_x.shape  # N: batch size, C: channels, H: height, W: width
            gram_x,gram_y=aux_features_x.view(N,C,H*W),aux_features_y.view(N,C,H*W)
            gram_x,gram_y=torch.bmm(gram_x,gram_x.transpose(1,2)),torch.bmm(gram_y,gram_y.transpose(1,2))
            return 1/(4*(H**2)*(W**2))*torch.sum((gram_x-gram_y)**2)

        l=0
        wl=1/(len(self.layers_style)-1)
        for i in range(len(self.layers_style)-1):
            l+=wl*loss_style_single(aux_features_x[i+1],aux_features_y[i+1])
        return l


    def load_image(self,path):
        return self.transform(Image.open(path).convert('RGB')) .to(device)


    def transfer(self,input_image_path,style_image_path,output_dir=os.path.join(ROOT_DIR,"generated"),wc=0.0001,ws=0.1,lr=1e-2):
        """
        Transfer the style from an input_image to a target image
        """
        """
        1. Read input and target images from their respective paths
        """
        input_image,style_image=self.load_image(input_image_path),self.load_image(style_image_path)
        input_image,style_image=input_image.unsqueeze(0),style_image.unsqueeze(0)
        """
        2. Generate the noise image:
        """

        noise=torch.randn (1, 3, INPUT_SIZE,INPUT_SIZE)

        # noise=input_image.clone()
        noise=noise.to(device)
        noise=noise+input_image.clone()
        noise.requires_grad=True

        """
        3.Configure optimizer
        """
        optimizer = Adam(params=[noise], lr=lr, betas=(0.9, 0.999))

        """
        4. Prepare generated
        """
        image_name=os.path.basename(input_image_path).replace(".jpg","").replace(".JPG","").replace(".png","").replace(".PNG","").replace(".jpeg","").replace(".JPEG","")
        style_name=os.path.basename(style_image_path).replace(".jpg","").replace(".JPG","").replace(".png","").replace(".PNG","").replace(".jpeg","").replace(".JPEG","")
        work=f"{image_name}_{style_name}"
        output_root=os.path.join(output_dir,work)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        shutil.copy(input_image_path,os.path.join(output_root,os.path.basename(input_image_path)))
        shutil.copy(style_image_path,os.path.join(output_root,os.path.basename(style_image_path)))

        logging.info("\n Transferring style from {} to {}".format(image_name,style_name))
        input_embeddings, input_aux = self(input_image)
        _, style_aux = self(style_image)
        #4. Optimize the noise image
        for step in range(self.num_iter):
            optimizer.zero_grad()
            noise_embeddings,noise_aux=self(noise)
            l_content=self.loss_content(input_aux,noise_aux)
            l_style=self.loss_style(style_aux,noise_aux)
            l_total=wc*l_content+ws*l_style
            l_total.backward()
            optimizer.step()
            if step%(self.num_iter/10)==0:
                logging.info("step:{}, l_content:{}, l_style:{}, l_total:{}".format(step,l_content,l_style,l_total))
                # output=self.upsampler.process(noise.detach().cpu().squeeze(0).permute(1,2,0))
                # Image.fromarray(output).convert("RGB").save(os.path.join(output_root,f"{work}.{step}.jpeg"))
                output=noise.detach().cpu().squeeze(0)
                save_image(output, os.path.join(output_root, f"{work}_{step}.jpeg"))
        save_image(output,os.path.join(output_root,f"{work}.jpeg"))
#