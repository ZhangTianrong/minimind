import argparse
import pathlib

DATASET_DIR = pathlib.Path(__file__).parent / "data"

def download_hf_dataset(destination_dir=DATASET_DIR):
    import huggingface_hub
    huggingface_hub.snapshot_download(
        "jingyaogong/minimind_dataset", 
        repo_type="dataset",
        local_dir=destination_dir,
        ignore_patterns=["*弃用*"]
    )

def setup_dataset_dir(link:pathlib.Path=None):
    if DATASET_DIR.exists():
        return
    if link is None:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
    else:
        link.mkdir(parents=True, exist_ok=True)
        DATASET_DIR.symlink_to(link, target_is_directory=True)

def main(link=None):
    setup_dataset_dir(link)
    download_hf_dataset()
   
parser = argparse.ArgumentParser("Download the MiniMind dataset.")
parser.add_argument("--setup-link", type=str, default=None, help="Create a directory at the specified path for the dataset directory to point to. This is useful when dataset needs to be stored elsewhere due to space constraints.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(pathlib.Path(args.setup_link) if args.setup_link else None)