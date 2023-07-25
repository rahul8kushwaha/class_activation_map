import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
import os


class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        layer = list(map(lambda a:a[1], torchvision.models.resnet18().named_children()))[:-2]
        self.get_feature_maps = nn.Sequential(*layer)
        self.class_weight = nn.Parameter(resnet.fc.weight.view(1000, 512, 1, 1))
        self.class_bias = nn.Parameter(resnet.fc.bias)
        self.get_CAM = nn.Conv2d(512, 1000, (1, 1), bias=True)
        self.get_CAM.weight = self.class_weight
        self.get_CAM.bias = self.class_bias
    def forward(self, PATH):
        x = self.read_image(PATH)
        feature_map = self.get_feature_maps(x)
        class_activation_maps = self.get_CAM(feature_map)
        return class_activation_maps, feature_map
    def read_image(self, filename):
        '''
        Function reads a image from the specified path and returns a tensor that can be feed to resnet
        '''
        input_image = Image.open(filename).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch
    
    
IMAGE_PATH = '/Users/rahulkushwaha/Downloads/good_images_for_paper/bird1.png'
model = CAM()
class_activation_maps, _ = model(IMAGE_PATH)
class_activation_maps.shape