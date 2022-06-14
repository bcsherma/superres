FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest
ENV WANDB_API_KEY $WANDB_API_KEY
ENV WANDB_PROJECT $WANDB_PROJECT