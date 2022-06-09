import wandb


def main():
    
    with wandb.init() as run:

        dataset = wandb.Artifact("div2k", type="image-dataset")

        dataset.add_reference("gs://wandb-superres-bucket/div2k/DIV2K_valid_HR/", name="valid_hr")
        dataset.add_reference("gs://wandb-superres-bucket/div2k/DIV2K_train_HR/", name="train_hr")
        dataset.add_reference("gs://wandb-superres-bucket/div2k/DIV2K_valid_LR_bicubic/X4", name="valid_lr")
        dataset.add_reference("gs://wandb-superres-bucket/div2k/DIV2K_train_LR_bicubic/X4", name="train_lr")

        run.log_artifact(dataset)

if __name__ == "__main__":  
    main()
