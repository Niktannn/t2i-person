## Overview

This repo contains source code for my bachelor's project: "High-resolution human image generation based on texutual description".

It generates images of a person whose description appears in given text. Image resolution is 1024x1024.

The service works as follows:
1) The service accepts a request containing text in Russian (e.g. "Блондинка в голубой кофте шла по улице.").
2) Then, using [Natasha](https://github.com/natasha/natasha), a description of characters (e.g. "блондинка") and their's features (e.g. "в голубой кофте") are extracted from the text.
3) Next, several random human images are generated using StyleGAN2 trained on custom dataset (model is taken from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)).
4) Gender classification is performed on every generated image using pretrained network from [age-and-gender](https://github.com/mowshon/age-and-gender). Then images of the corresponding gender are selected for the next stage.
5) For selected images several manipulations are performed according to textual descipriotns obtained at stage 2. For that "Latent optimization" and "Global direction" methods from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) are used.
6) On the obtained images of people, the background is cut out using [MODNet](https://github.com/ZHKKKe/MODNet/tree/revert-76-master).
7) Finally, images are encoded with base64 and sent back.

## Installation
### Download checkpoints
Download weights for SytleGAN2 model from [here](https://drive.google.com/file/d/12Ksgq5hiqyLY9bOyUTJb6cl894J6ukVw/view?usp=share_link) and put it into [models/StyleCLIP](models/StyleCLIP).

### Docker build
```sh
docker build -t t2i-person .
docker run -d -p 9008:9008 --name=[container-name] t2i-person
```
### Install packages
```sh
docker start [container-name]
docker attach [container-name]
pip install -v -e .
```
## Usage

### Send request
Running docker container sets up a server (on localhost:9008) that accepts requests in the following format:
- method: `t2i-person`
- Content-Type: `application/json`
- request format: `{"text": string}`
- answer format: `{“result”: list<list<string>>}` (list of base64-encoded images for every character)

### Manually
If you want to run server manually run [main.py](./main.py).