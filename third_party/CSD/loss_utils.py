import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import torch
from torch import nn
np.random.seed(0)


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return 
    

s=1
size = 224

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transforms_branch0 = transforms.Compose([
                transforms.Resize(size=size, interpolation=F.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

transforms_branch1 = transforms.Compose([
                transforms.RandomResizedCrop(size, interpolation=F.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.3), 
                transforms.RandomRotation(degrees=np.random.choice([0,90,180,270])),
                transforms.ToTensor(),
                normalize,
            ])

color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
transforms_branch2 = transforms.Compose([
                                        # transforms.RandomResizedCrop(size=size, interpolation=F.InterpolationMode.BICUBIC),
                                        transforms.Resize(size=size, interpolation=F.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(size),
                                        transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                            saturation=0.5,hue=0.1)
                                            ], p=0.6),
                                        transforms.RandomApply([transforms.RandomInvert(),transforms.RandomGrayscale(), transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 4))], p=0.8),
                                        # GaussianBlur(kernel_size=int(0.1 * size)),
                                        transforms.ToTensor(),
                                        normalize
                                        ])


class ContrastiveTransformations(object):

    def __init__(self, transforms_b0, transforms_b1,transforms_b2):
        self.transforms_b0 = transforms_b0
        self.transforms_b1 = transforms_b1
        self.transforms_b2 = transforms_b2
        
    def __call__(self, x):
        return [self.transforms_b0(x), self.transforms_b1(x), self.transforms_b2(x)]
