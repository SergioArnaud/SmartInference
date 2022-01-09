# importing module
import sys
import os
from boto3 import client
import yolov5
from torch.cuda import is_available as cuda_available
from tqdm import tqdm
import shutil
import argparse
from PIL import Image
# appending a path
sys.path.append('/home/ec2-user/SageMaker/la-comer/src/utils')
from file_handling import download_s3_file


def verify_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
        
def get_full_file_list(s3_client, bucket, prefix, Delimeter="/"):
    bucket_name = bucket
    s3_conn = s3_client
    if not prefix.endswith(Delimeter):
        prefix += Delimeter
    s3_result = s3_conn.list_objects_v2(Bucket=bucket_name,
                                        Prefix=prefix,
                                        Delimiter=Delimeter)

    if 'Contents' not in s3_result:
        return []

    file_list = []
    for key in s3_result['Contents']:
        file_list.append(key['Key'])
    # print(f"List count = {len(file_list)}")

    while s3_result['IsTruncated']:
        continuation_key = s3_result['NextContinuationToken']
        s3_result = s3_conn.list_objects_v2(Bucket=bucket_name,
                                            Prefix=prefix,
                                            Delimiter=Delimeter,
                                            ContinuationToken=continuation_key)
        for key in s3_result['Contents']:
            file_list.append(key['Key'])
    print(f"List count = {len(file_list)}")
    return file_list[1:]


class Image_saver:
    def __init__(self, img_path, i, base_path, yolo, confidence):
        """Saves and image, the label, and the bounding box.""" 
        self.img_path = img_path
        self.base_path = base_path
        self.save_path = os.path.join(self.base_path, 'images')
        self.label_path = os.path.join(self.base_path, 'labels')
        self.box_path = os.path.join(self.base_path, 'bounding-boxes')
        self.negative_path = os.path.join(self.base_path, 'negatives')
        self.yolo = yolo
        self.img = None
        self.label = None
        self.bounding_box = None
        self.i = i
        self.confidence = confidence
        
    def _verify_paths(self):
        for path in [self.base_path,
                     self.save_path,
                     self.label_path,
                     self.box_path,
                     self.negative_path]:
            verify_dir(path)
        self.save_path = os.path.join(self.save_path, f'{self.i}.jpg') 
        self.label_path = os.path.join(self.label_path, f'{self.i}.txt')
        self.box_path = os.path.join(self.box_path, f'{self.i}.txt')
        negative_i = len(os.listdir(self.negative_path))
        self.negative_path = os.path.join(self.negative_path, f'{negative_i}.jpg')
             
    def _read_img(self):
        self.img = Image.open(self.img_path)
        
    def _get_label(self):
        self.label = self.img_path.split('/')[-1].split('_')[0]
        
    def _create_bounding_box(self):
        r = self.yolo.predict(self.img)

        max_confidence = self.confidence
        x1, y1, x2, y2 = -1, -1, -1, -1
        for i, (x1, y1, x2, y2, confidence, _class) in enumerate(r.xyxy[0]):
            if confidence > max_confidence:
                x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
                max_confidence = confidence

        if x1 != -1:
            self.bounding_box = f"{x1} {y1} {x2} {y2}"
                
    def save(self):
        self._verify_paths()
        self._read_img()
        self._get_label()
        self._create_bounding_box()
        
        if self.bounding_box is None:
            self.img.convert('RGB').save(self.negative_path)
            os.remove(self.img_path)
            return self.i
        self.img.save(self.save_path)
        with open(self.label_path, 'w') as file:
            file.write(self.label)
        with open(self.box_path, 'w') as file:
            file.write(self.bounding_box)
        os.remove(self.img_path)
        return self.i + 1
        

class ComerS3Dataset:
    def __init__(self, s3_bucket,s3_prefix, base_path, yolo_path, yolos3 = True, confidence = 0.80, device=None, self):
        self.client = client('s3')
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.base_path = base_path
        self.keys = None
        self.new_names = None
        self.momentary_path = os.path.join(base_path, 's3-downloads')
        self._verify_paths()
        self.confidence = confidence
        if device is None:
            device = 'cuda' if cuda_available() else 'cpu'
        if yolos3:
            new_yolo_path = os.path.join(self.momentary_path, yolo_path.split('/')[-1])
            download_s3_file(s3_bucket, yolo_path, new_yolo_path)
            self.yolo = yolov5.YOLOv5(new_yolo_path, device=device)
        else:
            self.yolo = yolov5.YOLOv5(yolo_path, device=device)
        
    def _get_keys(self):
        self.keys = get_full_file_list(self.client, self.s3_bucket, self.s3_prefix)
        
    def _get_new_names(self):
        new_names = [f"{self.momentary_path}/{key.split('/')[-1]}" for key in self.keys]
        self.new_names = new_names
        
    def _verify_paths(self):
        verify_dir(self.base_path)
        verify_dir(self.momentary_path)
    
    def _setup(self):
        self._get_keys()
        self._get_new_names()
        self._verify_paths()
    
    def _download_item(self, idx):
        download_s3_file(self.s3_bucket, self.keys[idx], self.new_names[idx])
        return self.keys[idx], self.new_names[idx]
    
    def build(self):
        """Builds the dataset in the porvided location in init."""        
        self._setup()
        i = 0
        for key_idx in tqdm(range(len(self.keys))):
            key, img_path = self._download_item(key_idx)
            i = Image_saver(img_path, i, self.base_path, self.yolo, self.confidence).save()
        shutil.rmtree(self.momentary_path)
            
            
def main(args=None):
    if args is not None:
        base_path = args.base_path  # 's3-images-3'
        verify_dir(base_path)
        s3_bucket = args.bucket  # 'la-comer'
        s3_prefix = args.s3_prefix  # 'data/comer-images-final-3/'
        yolo_path = args.yolo_path
        ComerS3Dataset(s3_bucket,
                       s3_prefix,
                       base_path,
                       yolo_path,
                       args.yolos3,
                       confidence=args.confidence,
                       device=args.device).build()
    else:
        ComerS3Dataset(
            'la-comer', 
            'data/comer-images-final-3/', 
            "la-comer/src/data/comer-images/dataset",
            'train/yolov5_runs/weights/5m_full_filtered_augmentated_negative.pt',
            True).build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images with potential barcodes from and s3 bucket"
        )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/ec2-user/SageMaker/la-comer/src/data/comer-images/dataset",
        help="path to la-comer images base",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default='la-comer',
        help="S3 bucket",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default='data/comer-images-final-3/',
        help="s3-location of images within specified bucket in --bucket",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use in yolo",
    )
    parser.add_argument(
        "--yolo_path",
        type=str,
        default='train/yolov5_runs/weights/5m_full_filtered_augmentated_negative.pt',
        help="path to yolo, if local path provided must specify via --yolos3 False",
    )
    parser.add_argument(
        "--yolos3",
        type=bool,
        default=True,
        help="wether to treat yolo_path as s3 key or local path",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Min Confidence Threshold for Yolo",
    )
    args = parser.parse_args()

    main(args)
    