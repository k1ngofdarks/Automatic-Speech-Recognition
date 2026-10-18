from pathlib import Path

import huggingface_hub as hf
import hydra


@hydra.main(
    version_base=None, config_path="../src/configs", config_name="download_resources"
)
def main(config):
    for resource in config.resources:
        resource_path = Path(resource.local_dir) / resource.filename
        if resource_path.is_file():
            print(f"{resource_path} already exists. Skipping download.")
            continue

        print(f"Downloading {resource.filename} from {config.repo_id}...")
        path = hf.hf_hub_download(
            repo_id=config.repo_id,
            filename=resource.filename,
            local_dir=resource.local_dir,
        )
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
