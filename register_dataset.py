import argparse
import wandb


def parse_args():
    parser = argparse.ArgumentParser("register_dataset.py")
    parser.add_argument("bucket", type=str)
    parser.add_argument("scale_factor", type=int, default=4)
    return parser.parse_args()


def main():

    args = parse_args()

    with wandb.init(job_type="register_data", config=args) as run:

        config = run.config
        dataset = wandb.Artifact(
            "div2k",
            type="image-dataset",
            metadata=dict(scale_factor=config.scale_factor),
        )

        dataset.add_reference(
            f"gs://{config.bucket}/div2k/DIV2K_valid_HR/", name="valid_hr"
        )
        dataset.add_reference(
            f"gs://{config.bucket}/div2k/DIV2K_train_HR/", name="train_hr"
        )
        dataset.add_reference(
            f"gs://{config.bucket}/div2k/DIV2K_valid_LR_bicubic/X{config.scale_factor}",
            name="valid_lr",
        )
        dataset.add_reference(
            f"gs://{config.bucket}/div2k/DIV2K_train_LR_bicubic/X{config.scale_factor}",
            name="train_lr",
        )
        run.log_artifact(dataset)

if __name__ == "__main__":
    main()
