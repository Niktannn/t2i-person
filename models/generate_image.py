import os
import random
import torch
import base64

from io import BytesIO
from argparse import Namespace
from PIL import Image

from .StyleCLIP.optimization.run_optimization import main
from .StyleCLIP.gender_search import find_gender
from .StyleCLIP_pytorch.global_direction import generate_gd
from .MODNet.demo.image_matting.Inference_with_ONNX.inference_onnx import matte

torch.manual_seed(1)

MAX_DEPTH = 2
MAX_LAYER_SIZE = 3
GENDER_SAMPLES_NUM = 2

args = {
    "ckpt": "models/StyleCLIP/network-snapshot-000200.pt",
    "stylegan_size": 1024,
    "lr_rampup": 0.15,
    "lr": 0.003,
    "step": 80,
    "mode": 'edit',
    "l2_lambda": 0.00005,
    "id_lambda": 0.0,
    'work_in_stylespace': False,
    "latent_path": None,
    "truncation": 0.36,
    "save_intermediate_image_every": 0,
    "results_dir": "results",
    "ir_se50_weights": "checkpoints/model_ir_se50.pth",
    "latent_save_path": None,
    "image_save_name": None,
    "gender": None
}


def matte_image(im):
    (low, high) = (-1.0, 1.0)

    image = im[0]
    image.clamp_(min=low, max=high)
    image.sub_(low).div_(max(high - low, 1e-5))
    image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    mat = Image.fromarray(matte(image))

    im_pil = Image.fromarray(image)
    im_pil.putalpha(mat)
    im_pil.convert('RGB')
    return im_pil


def encode(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
    return img_str


def generate_image(desc, output_latent_name,
                   global_direction=False,
                   edit=False, input_latent_name=None,
                   complex_description=False,
                   gender=None, gender_samples_num=GENDER_SAMPLES_NUM):
    if global_direction:
        assert input_latent_name is not None
        return generate_gd(f'models/StyleCLIP/results/{input_latent_name}.pt', desc)
    else:
        cur_args = args.copy()
        cur_args['description'] = desc
        if edit:
            cur_args['latent_path'] = f'models/StyleCLIP/results/{input_latent_name}.pt'
        cur_args['latent_save_path'] = f'models/StyleCLIP/results/{output_latent_name}.pt'
        cur_args['image_save_name'] = output_latent_name
        if complex_description:
            cur_args['l2_lambda'] = cur_args['l2_lambda'] / 2
            cur_args['lr'] = cur_args['lr'] * 1.4
            cur_args['step'] = int(cur_args['step'] * 1.5)

        if gender:
            find_gender(Namespace(**cur_args), gender, latent_name=output_latent_name, gender_samples_num=gender_samples_num)
            return
        else:
            result = main(Namespace(**cur_args))
            return result.detach().cpu()


def encode_and_save(res, name):
    matted = matte_image(res)
    matted.save(name)
    encoded = encode(matted)
    return encoded


def update_name(start, next):
    return f'{start}_{next}'


def get_load_save_names(state_idx, layer_idx, features_for_layers, features, name_start, cur_feature):
    load_name = ''
    cur_v_num = state_idx
    for prev_layer_num in range(layer_idx):
        cur_layer_num = layer_idx - 1 - prev_layer_num
        if not len(features_for_layers[cur_layer_num]) == len(features_for_layers[cur_layer_num + 1]):
            cur_v_num //= 2
        if len(load_name) == 0:
            load_name = features[features_for_layers[cur_layer_num][cur_v_num]][0]
        else:
            load_name = update_name(features[features_for_layers[cur_layer_num][cur_v_num]][0], load_name)
    load_name = update_name(name_start, load_name)
    save_name = update_name(load_name, cur_feature)
    return load_name, save_name


def generate_images(descriptions):
    print("Descriptions: ", descriptions)

    dirs_to_clear = ['result', 'results', 'models/StyleCLIP/results']
    for dir_to_clear in dirs_to_clear:
        for f in os.listdir(dir_to_clear):
            os.remove(os.path.join(dir_to_clear, f))

    results = []
    for desc_i, description in enumerate(descriptions):
        cur_person_results = []
        gender = 'male' if description['gen'] == 'Masc' else 'female'
        init_name_start = f'{desc_i}_{gender}'
        features = [description['obj']] + description['desc']
        n = len(features)

        generate_image("", init_name_start, gender=gender, gender_samples_num = GENDER_SAMPLES_NUM)
        for gender_sample_idx in range(GENDER_SAMPLES_NUM):
            name_start = update_name(init_name_start, str(gender_sample_idx))

            combined_desc = " ".join([f for f, _ in features])
            save_name = update_name(name_start, combined_desc)

            res = generate_image(combined_desc, f'combined_{save_name}',
                                 edit=True, input_latent_name=name_start, complex_description=True)
            cur_person_results.append(encode_and_save(res, f'result/result_combined_{save_name}.png'))

            depth = min(MAX_DEPTH, n - 1)
            layer_sz = min(MAX_LAYER_SIZE, n)
            features_for_layers = [random.sample(range(n), layer_sz)]

            for feature_idx in features_for_layers[0]:
                feature, is_gd = features[feature_idx]
                save_name = update_name(name_start, feature)

                if n == 1:
                    res = generate_image(feature, None, global_direction=True, input_latent_name=name_start)
                    cur_person_results.append(encode_and_save(res, f'result/result_{save_name}.png'))
                else:
                    generate_image(feature, save_name, global_direction=is_gd,
                                         edit=True, input_latent_name=name_start)

            for d in range(1, depth + 1):
                cur_layer = []
                for state_idx in range(layer_sz * (2 ** d)):
                    not_used = list(range(n))
                    prev_state = state_idx
                    for layer_idx in range(d):
                        prev_state //= 2
                        not_used.remove(features_for_layers[d - layer_idx - 1][prev_state])
                    if state_idx % 2 == 1:
                        not_used.remove(cur_layer[-1])
                    if len(not_used) == 0:
                        continue
                    cur_layer.append(random.choice(not_used))
                features_for_layers.append(cur_layer)

            for layer_idx in range(1, len(features_for_layers)):
                for state_idx, feature_idx in enumerate(features_for_layers[layer_idx]):
                    cur_feature, is_gd = features[feature_idx]

                    load_name, save_name = get_load_save_names(state_idx, layer_idx, features_for_layers, features,
                                                               name_start, cur_feature)

                    if layer_idx == len(features_for_layers) - 1:
                        res = generate_image(cur_feature, None, global_direction=True, input_latent_name=load_name)
                        cur_person_results.append(encode_and_save(res, f'result/result_{save_name}.png'))
                    else:
                        generate_image(cur_feature, save_name, global_direction=is_gd, edit=True,
                                       input_latent_name=load_name)

        results.append(cur_person_results)
    return results
