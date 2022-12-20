# import os
# import random
# import torch
#
# from .StyleCLIP.optimization.run_optimization import main
#
# from argparse import Namespace
# from itertools import product
#
# from datetime import datetime
#
# # random.seed(seed)
#
# experiment_type = 'edit'
# latent_path = None
# id_lambda = 0.00
# stylespace = False
# create_video = False
# use_seed = True
#
# step_vs         = [75, 250]                         #   75 250
# l2_vs           = [0.00125, 0.00025, 0.00005]       #   0.00125 0.00025 0.00005
# stylespace_v    = False                             #   False
# lr_vs           = [0.075, 0.015, 0.003]             #   0.075 0.015 0.003
# rampup_vs       = [0.3, 0.1, 0.03]                  #   0.3 0.1 0.03
# trunc_v         = 0.45                              #   0.45
#
# params = list(product(step_vs, l2_vs, lr_vs, rampup_vs))
#
# max_depth = 2
# max_layer_size = 3
#
# args = {}
#
#
# def generate_image(desc, output_latent_name, desc_add, edit=False, input_latent_name=None):
#     seed = int(datetime.now().timestamp()) % 128 + 192
#     print("seed:", seed)
#     if use_seed:
#         torch.manual_seed(seed)
#
#     cur_args = args.copy()
#     cur_args['description'] = desc
#     if edit:
#         cur_args['latent_path'] = f'models/StyleCLIP/results/{input_latent_name}.pt'
#     cur_args['latent_save_path'] = f'models/StyleCLIP/results/{output_latent_name}.pt'
#     cur_args['image_save_name'] = output_latent_name + desc_add
#
#     result = main(Namespace(**cur_args))
#     return result.detach().cpu()
#
#
# def search(descriptions):
#     dirs_to_clear = ['result', 'results', 'models/StyleCLIP/results']
#     for dir_to_clear in dirs_to_clear:
#         for f in os.listdir(dir_to_clear):
#             os.remove(os.path.join(dir_to_clear, f))
#
#     for (step_v, l2_v, lr_v, rampup_v) in params:
#         global args
#         args = {
#             # "description": description,
#             "ckpt": "models/StyleCLIP/network-snapshot-000200.pt",
#             "stylegan_size": 1024,
#             "lr_rampup": rampup_v,
#             "lr": lr_v,
#             "step": step_v,
#             "mode": experiment_type,
#             "l2_lambda": l2_v,
#             "id_lambda": id_lambda,
#             'work_in_stylespace': stylespace_v,
#             "latent_path": None,
#             "truncation": trunc_v,
#             "save_intermediate_image_every": 0,
#             "results_dir": "results",
#             "ir_se50_weights": "checkpoints/model_ir_se50.pth",
#             "latent_save_path": None,
#             "image_save_name": None
#         }
#         generate_images(descriptions, f"_st-{step_v}_l2-{l2_v}_lr-{lr_v}_rp-{rampup_v}")
#
#
# def generate_images(descriptions, desc_add):
#     print("**", descriptions, "**")
#
#     dirs_to_clear = ['models/StyleCLIP/results']
#     for dir_to_clear in dirs_to_clear:
#         for f in os.listdir(dir_to_clear):
#             os.remove(os.path.join(dir_to_clear, f))
#
#     for desc_i, description in enumerate(descriptions):
#         init_desc = ('male' if description['gen'] == 'Masc' else 'female') + ' ' + description['obj']
#         descs = [init_desc] + description['desc']
#         n = len(descs)
#
#         combined_desc = ''
#         for desc in descs:
#             combined_desc += (desc + ' ')
#         generate_image(combined_desc, f'combined_{combined_desc}', desc_add)
#
#         depth = min(max_depth, n - 1)
#         layer_sz = min(max_layer_size, n)
#         layers = [random.sample(range(n), layer_sz)]
#
#         for desc_num in layers[0]:
#             generate_image(descs[desc_num], descs[desc_num], desc_add)
#
#         for d in range(depth):
#             cur_layer = []
#             for v_num in range(layer_sz * (2 ** (d + 1))):
#                 not_used = list(range(n))
#                 cur_v_num = v_num
#                 for layer_num in range(d + 1):
#                     cur_v_num //= 2
#                     not_used.remove(layers[d - layer_num][cur_v_num])
#                 if v_num % 2 == 1:
#                     not_used.remove(cur_layer[-1])
#                 if len(not_used) == 0:
#                     continue
#                 cur_layer.append(random.choice(not_used))
#             layers.append(cur_layer)
#
#         for layer_num in range(1, len(layers)):
#             for v_num, desc_num in enumerate(layers[layer_num]):
#                 cur_desc = descs[desc_num]
#
#                 load_name = ''
#                 cur_v_num = v_num
#                 for prev_layer_num in range(layer_num):
#                     cur_layer_num = layer_num - 1 - prev_layer_num
#                     if not len(layers[cur_layer_num]) == len(layers[cur_layer_num + 1]):
#                         cur_v_num //= 2
#                     if len(load_name) == 0:
#                         load_name = descs[layers[cur_layer_num][cur_v_num]]
#                     else:
#                         load_name = descs[layers[cur_layer_num][cur_v_num]] + '_' + load_name
#
#                 save_name = load_name + '_' + cur_desc
#                 generate_image(cur_desc, save_name, desc_add, True, load_name)