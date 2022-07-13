FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest
COPY . /src
WORKDIR /src
RUN pip install -r requirements.txt