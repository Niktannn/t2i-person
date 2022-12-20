import os
import clip
import numpy as np
import PIL.Image
import torch
import torchvision

from torchvision.transforms import ToPILImage

from .embedding import get_delta_t
from .manipulator import Manipulator
from .mapper import get_delta_s
from .wrapper import Generator

templates = [
    'a bad photo of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a dark photo of a {}.',
    'graffiti of the {}.',
]


def generate_gd(latent_path, target_description, source_description='a person'):
    # GPU device
    device = torch.device('cuda:0')
    # pretrained generator
    ckpt = 'models/StyleCLIP_pytorch/pretrained/network-snapshot-000200.pkl'
    G = Generator(ckpt, device)
    # CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    # global image direction
    fs3 = np.load('models/StyleCLIP_pytorch/tensor/fs3network-snapshot-000200.npy')
    manipulator = Manipulator(G, device)

    manipulator.map_latent(latent_path)

    # beta_threshold : Determines the degree of disentanglement, # channels manipulated
    beta_threshold = 0.085

    classnames = [source_description, target_description]
    # get delta_t in CLIP text space
    delta_t = get_delta_t(classnames, model)
    # get delta_s in global image directions and text directions that satisfy beta threshold
    left_beta = 0.08
    right_beta = 0.15
    eps = 0.0001
    target_num_channel_range = (700, 1100)
    num_channel = 0

    while (not (target_num_channel_range[0] <= num_channel <= target_num_channel_range[1])
           and right_beta - left_beta > eps):
        beta_threshold = (left_beta + right_beta) / 2
        delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta_threshold)
        if num_channel < target_num_channel_range[0]:
            right_beta = beta_threshold
        elif num_channel > target_num_channel_range[1]:
            left_beta = beta_threshold
        else:
            break
    print(f'{num_channel} channels will be manipulated under the beta threshold {beta_threshold}')

    # alpha_threshold : Determines the degree of manipulation
    # lst_alpha = [-3.0, -1.5, 0.0, 1.5, 3.0]
    lst_alpha = [-1.8]
    manipulator.set_alpha(lst_alpha)

    # manipulate styles
    styles = manipulator.manipulate(delta_s)

    # synthesis images from manipulated styles
    all_imgs = manipulator.synthesis_from_styles(styles, 0, manipulator.num_images)

    # print(f'SHAPE:{all_imgs.shape}')

    # lst = []
    #
    # for res in all_imgs:
    #     lst.append(res[0])

    # result_image = ToPILImage()(torchvision.utils.make_grid(lst,
    #                                                         normalize=True, scale_each=True, range=(-1, 1), padding=0))
    # latent_name = os.path.split(latent_path)[-1].split('.')[0]
    # result_image.save(f'result/GD_{latent_name}_from_{source_description}_to_{target_description}.png')
    return all_imgs[0]

