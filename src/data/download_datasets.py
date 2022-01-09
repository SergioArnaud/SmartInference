"""
This file has all the function that download datasets of barcodes.
Every dataset should be saved in https://s3.console.aws.amazon.com/s3/buckets/la-comer?prefix=data/
in a folder with the name of the dataset source

It should be run only once but it's in here for reproducibility
"""

import os
import gdown
from utils.file_handling import upload_to_s3
import git

repo = git.Repo(".", search_parent_directories=True)
base_path = f"{repo.working_tree_dir}/src"
os.chdir(base_path)


def download_barcode_detection_benchmark():
    """
    If you go to the drive and click share you will get a link, it's important to put
    that link in us (drive.google.com/file/d/Afah... -> drive.google.com/uc?id=Afah)
    """

    path = "data/barcode_detection_benchmark"
    os.makedirs(path, exist_ok=True)

    url_real = "https://drive.google.com/uc?id=1kXhWySN-WagEnxhw8LyCpx8CjDOfndnT"
    url_synth = "https://drive.google.com/uc?id=10Rz0zK_3LGUDIMXH8zQrokskWb5eWN-t"

    output_real = f"{path}/barcode_real_dataset.zip"
    output_synth = f"{path}/barcode_synth_dataset.zip"

    gdown.download(url_real, output_real, quiet=False)
    gdown.download(url_synth, output_synth, quiet=False)

    upload_to_s3(
        "la-comer", output_real, output_real.replace("/barcode_real_dataset.zip", "")
    )
    upload_to_s3(
        "la-comer", output_synth, output_synth.replace("/barcode_synth_dataset.zip", "")
    )


if __name__ == "__main__":
    download_barcode_detection_benchmark()
