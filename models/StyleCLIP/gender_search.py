import os

import torch
import torchvision
from torchvision.transforms import ToPILImage

from models.gender import predict_gender

from .models.stylegan2.model import Generator
from .utils import ensure_checkpoint_exists


confidence_threshold = 0.6

# def find_gender(args, gender):
#     ensure_checkpoint_exists(args.ckpt)
#     os.makedirs(args.results_dir, exist_ok=True)
#
#     g_ema = Generator(args.stylegan_size, 512, 8)
#     g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
#     g_ema.eval()
#     g_ema = g_ema.cuda()
#     mean_latent = g_ema.mean_latent(4096)
#
#     max_target_confidence = 0.0
#     min_opposite_confidence = 100.0
#
#     found_target_gender = False
#     best_target_img = None
#     best_target_latent = None
#     best_opposite_img = None
#     best_opposite_latent = None
#
#     for j in range(gender_samples_num):
#         latent_code_init_not_trunc = torch.randn(1, 512).cuda()
#         with torch.no_grad():
#             _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
#                                            truncation=args.truncation, truncation_latent=mean_latent)
#             img, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
#             pil_img = ToPILImage()(torchvision.utils.make_grid(img, normalize=True, scale_each=True, range=(-1, 1),
#                                                                padding=0))
#             res = predict_gender.get_gender(pil_img)
#             predicted_gender = res['value']
#             confidence = float(res['confidence'])
#             if predicted_gender == gender:
#                 if not found_target_gender:
#                     found_target_gender = True
#                     max_target_confidence = confidence
#                     best_target_img = img.detach().clone()
#                     best_target_latent = latent_code_init.detach().clone()
#                 elif confidence > max_target_confidence:
#                     max_target_confidence = confidence
#                     best_target_img = img.detach().clone()
#                     best_target_latent = latent_code_init.detach().clone()
#             elif confidence < min_opposite_confidence:
#                 min_opposite_confidence = confidence
#                 best_opposite_img = img.detach().clone()
#                 best_opposite_latent = latent_code_init.detach().clone()
#
#             # torchvision.utils.save_image(img, f"{args.results_dir}/{args.image_save_name}_{j}_{predicted_gender}_{confidence}.png",
#             #                              normalize=True, range=(-1, 1))
#
#     # if not (args.image_save_name is None):
#     #     torchvision.utils.save_image(best_target_img if found_target_gender else best_opposite_img,
#     #                                  f"{args.results_dir}/{args.image_save_name}_{max_target_confidence}.png",
#     #                                  normalize=True, range=(-1, 1))
#
#     if not (args.latent_save_path is None):
#         torch.save(best_target_latent if found_target_gender else best_opposite_latent, args.latent_save_path)
#
#     return best_target_img if found_target_gender else best_opposite_img


def find_gender(args, gender, latent_name, gender_samples_num=3):
    ensure_checkpoint_exists(args.ckpt)
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    found_samples = 0
    result_images = []
    result_latents = []

    while found_samples != gender_samples_num:
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                           truncation=args.truncation, truncation_latent=mean_latent)
            img, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
            pil_img = ToPILImage()(torchvision.utils.make_grid(img, normalize=True, scale_each=True, range=(-1, 1),
                                                               padding=0))
            res = predict_gender.get_gender(pil_img)
            predicted_gender = res['value']
            confidence = float(res['confidence'])
            if predicted_gender == gender and confidence >= confidence_threshold:
                found_samples+=1
                result_images.append(img.detach().clone())
                result_latents.append(latent_code_init.detach().clone())


            # torchvision.utils.save_image(img, f"{args.results_dir}/{args.image_save_name}_{j}_{predicted_gender}_{confidence}.png",
            #                              normalize=True, range=(-1, 1))

    # if not (args.image_save_name is None):
    #     torchvision.utils.save_image(best_target_img if found_target_gender else best_opposite_img,
    #                                  f"{args.results_dir}/{args.image_save_name}_{max_target_confidence}.png",
    #                                  normalize=True, range=(-1, 1))

    for i, img in enumerate(result_images):
        torchvision.utils.save_image(img,f"{args.results_dir}/{args.image_save_name}_{i}.png",
                                         normalize=True, range=(-1, 1))

    for i, latent in enumerate(result_latents):
        torch.save(latent, f'models/StyleCLIP/results/{latent_name}_{i}.pt')

    return result_images
