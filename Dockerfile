#FROM python:3.7
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y apt-utils python3 python3-pip ffmpeg libsm6 libxext6 python3-venv git libxtst-dev libpng++-dev libjpeg-dev
#RUN pip3 install virtualenv

ENV VIRTUAL_ENV=/inference-api/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements.txt /inference-api/requirements.txt

#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip3 install -U pip wheel cmake ninja
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install -r /inference-api/requirements.txt
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install git+https://github.com/mowshon/age-and-gender.git
RUN pip3 install cython --no-cache-dir

#COPY ./models.tar.gz /inference-api/
#RUN tar -zxf /inference-api/models.tar.gz --directory /inference-api
#RUN rm /inference-api/models.tar.gz

#COPY ./inference-api/* /inference-api/inference-api/
COPY ./models /inference-api/models
COPY ./checkpoints /inference-api/checkpoints
COPY ./result /inference-api/result
COPY ./results /inference-api/results
COPY server.py /inference-api/server.py
COPY ./config.yml /inference-api/config.yml


WORKDIR /inference-api/

EXPOSE 8080

CMD [ "python", "/inference-api/server.py" ]