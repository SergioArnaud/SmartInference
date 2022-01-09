import traceback
import whatimage
from SyntheticGenerator import SyntheticGenerator
import yolov5
from tqdm import tqdm
import os
import json
import git
import sys
import gdown
import pyheif
from PIL import Image
from random import sample, uniform
import cv2
from augmenter import Augmenter
import random
import shutil

repo = git.Repo(".", search_parent_directories=True)
base_path = f"{repo.working_tree_dir}/src"
sys.path.append(base_path)
from utils.file_handling import download_s3_file, unzip # noqa E402


random.seed(420)


def convert_to_jpg(path, image_name):
    extention = image_name.split(".")[-1]

    with open(f"{path}/{image_name}", "rb") as f:
        data = f.read()
    real_extention = whatimage.identify_image(data)

    if real_extention in ["HEIC", "heic"]:
        file = pyheif.read(f"{path}/{image_name}")
        image = Image.frombytes(
            file.mode,
            file.size,
            file.data,
            "raw",
            file.mode,
            file.stride,
        )

    else:
        file = Image.open(f"{path}/{image_name}")
        image = file.convert("RGB")

    new_name = f"{path}/{image_name.replace(extention, 'jpg')}"
    image.save(new_name)
    return new_name


def get_ROI(image, xyxy, margin_pct):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))

    margin = int((x2 - x1) * margin_pct / 100)

    x1 = int(float(x1 - margin)) if x1 - margin >= 0 else int(float(x1))
    y1 = int(float(y1 - margin)) if y1 - margin >= 0 else int(float(y1))
    x2 = int(float(x2 + margin)) if x2 + margin <= image.shape[1] else int(float(x2))
    y2 = int(float(y2 + margin)) if y2 + margin <= image.shape[0] else int(float(y2))

    return image[y1:y2, x1:x2]


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


class BarcodeDatasetPreparer:
    def __init__(self, dataset_path, yolo_path, num_synth=10, test_size=1500, device = 'cuda', 
                 synth_scaling_factor = 2, real_scaling_factor = 5, prob_save_original_synth = .5):

        self.dataset_path = dataset_path
        self.bucket_name = "la-comer"
        self.num_synth = num_synth
        self.test_size = test_size
        self.synth_scaling_factor = synth_scaling_factor
        self.real_scaling_factor = real_scaling_factor
        self.real_folder = "unprocessed_real"
        self.synth_folder = "unprocessed_synth"
        self.prob_save_original_synth = prob_save_original_synth
        self.yolo = yolov5.YOLOv5(yolo_path, device=device)

        # Inicializamos folder reales y sinteticos
        self.init_folders()
        self.get_comer_barcodes()
        
    def process(self):

        # Obtenemos imagenes reales de diversas fuentes (paper comer y divers)
        print("Getting Real Images")
        self.get_real_images()

        # Generamos sintéticas
        self.generate_synth()

        # Añadimos bounding box a todos los reales
        self.add_missing_bounding_boxes()
        
        # Quita del folder unprocessed_real el conujunto de imágenes  de test
        self.create_test_real()

        # itera cada imagen real, la cropea, aumenta y añade al folder final 
        self.create_train_real()

        # itera cada imagen synth, aumenta y añade al folder final
        self.create_train_synth()

    def init_folders(self):
        create_folder(self.dataset_path)
        create_folder(self.dataset_path + "/train")
        create_folder(self.dataset_path + "/test")
        create_folder(self.dataset_path + "/train/images")
        create_folder(self.dataset_path + "/train/labels")
        create_folder(self.dataset_path + "/test/images")
        create_folder(self.dataset_path + "/test/labels")

        create_folder(self.real_folder)
        create_folder(self.real_folder + "/images")
        create_folder(self.real_folder + "/labels")
        create_folder(self.real_folder + "/bounding_boxes")

        create_folder(self.synth_folder)
        create_folder(self.synth_folder + "/images")
        create_folder(self.synth_folder + "/labels")
    
    def get_comer_barcodes(self):
        def unique(arr):
            return list(set(arr))
        download_s3_file('la-comer', 'data/comer-data/data.json', 'comer_products.json')
        with open('comer_products.json',) as f:
            data = json.load(f)
        barcodes = [dict_['ean'] for dict_ in data]
        ean13 = [code for code in barcodes if self.is_ean13(code)]
    
        self.comer_barcodes = ean13
        self.countries_barcodes = unique([barcode[:3] for barcode in ean13])
        self.manufacturers_barcodes = unique([barcode[3:7] for barcode in ean13])
        
    def generate_synth(self):
        SG = SyntheticGenerator(
            self.synth_folder, 
            self.num_synth, 
            barcode_list=self.comer_barcodes,
            country_list=self.countries_barcodes,
            manufacturer_list=self.manufacturers_barcodes
        )
        SG.generate()

    def get_bounding_boxes(self, path):
        bounding_boxes = {}
        with open(path, "r") as f:
            for line in f:
                barcode, a, b, c, d, cl = line.split(" ")
                bounding_boxes[barcode] = f"{a} {b} {c} {d}"
        return bounding_boxes

    def write_image_label_box(self, image_path, image_name, barcode, bounding_box=None):
        path = self.real_folder
        os.rename(image_path, f"{path}/images/{image_name}.jpg")

        with open(f"{self.real_folder}/labels/{image_name}.txt", "w") as file:
            file.write(barcode)

        if bounding_box:
            with open(
                f"{self.real_folder}/bounding_boxes/{image_name}.txt", "w"
            ) as file:
                file.write(bounding_box)

    def is_ean13(self, barcode):
        
        if len(barcode) == 13:
            try:
                int(barcode)
                return True
            except:
                return False
        else:
            return False
        
        
    def process_bardode_bb_datasets(self, folder):
        path = f"barcode_bb/barcode_bb/{folder}"
        images = os.listdir(f"{path}")
        bounding_boxes = self.get_bounding_boxes(f"{path}.txt")

        for k, image in tqdm(
            enumerate(images), total=len(images), desc=f"barcode bb {folder} parsing"
        ):
            image_path = convert_to_jpg(path, image)
            bounding_box = bounding_boxes[image]
            barcode = image.split(".")[0].split("-")[0].split("_")[0].split(' ')[0]
            if self.is_ean13(barcode):
                self.write_image_label_box(
                    image_path, f"{k}-bb-{folder}", barcode, bounding_box
                )

    def get_real_paper(self):
        path_s3 = "data/smart_inference_data"

        key = "barcode_bb.zip"
        download_s3_file(self.bucket_name, f"{path_s3}/{key}", key)
        unzip(key)

        self.process_bardode_bb_datasets("rec")
        self.process_bardode_bb_datasets("single_test")

    def get_real_detection_benchmark(self):

        path_s3 = "data/smart_inference_data"
        key = "Real_EAN13.zip"
        download_s3_file(self.bucket_name, f"{path_s3}/{key}", key)
        unzip(key)

        image_path = "Real_EAN13/Real_EAN13/Image/"
        label_path = "Real_EAN13/Real_EAN13/Label/"
        images = os.listdir(image_path)

        for k, image in enumerate(images):
            label = image.replace("jpg", "txt")
            image_name = f"{k}-bdb-ean13"
            os.rename(image_path + image, f"{self.real_folder}/images/{image_name}.jpg")
            os.rename(label_path + label, f"{self.real_folder}/labels/{image_name}.txt")

    def get_real_divers(self):
        url = "https://drive.google.com/uc?id=1BJONvGlXB-_wYJv7WpOU-6ijarKRWPyA"
        output = "barcodes-ean13.zip"
        gdown.download(url, output, quiet=False)
        unzip(output)

        base_path = "barcodes-ean13/barcodes-ean-13/"
        for diver in tqdm(os.listdir(base_path), desc="Processing diver Images"):
            path = base_path + diver
            for k, img in enumerate(os.listdir(path)):
                image_path = convert_to_jpg(path, img)
                barcode = img.split(".")[0].split("-")[0].split("_")[0].split(' ')[0]
                image_name = f"{k}-{diver}-ean13"
                if self.is_ean13(barcode):
                    self.write_image_label_box(image_path, image_name, barcode)

                    
    def get_real_images_comer(self):
        base_path = '../comer-images/dataset/'
        
        images_path = f'{base_path}/images/'
        labels_path = f'{base_path}/labels/'
        bounding_boxes_path = f'{base_path}/bounding-boxes/'
        
        for image_name in os.listdir(images_path):
            if '.jpg' not in image_name:
                continue
                
            label_name = image_name.replace('jpg','txt')
            
            with open(labels_path + label_name) as f:
                barcode = f.read()
                
            if self.is_ean13(str(barcode)):
                shutil.copyfile(images_path + image_name, f"{self.real_folder}/images/comer_{image_name}")
                shutil.copyfile(labels_path + label_name, f"{self.real_folder}/labels/comer_{label_name}")
                shutil.copyfile(bounding_boxes_path + label_name, f"{self.real_folder}/bounding_boxes/comer_{label_name}")
    
    def get_real_images(self):

        print("Getting Images from the paper")
        self.get_real_paper()

        # print('Getting Images from the barcode detection benchmark')
        # self.get_real_detection_benchmark()
        
        print('Getting Images from comer')
        self.get_real_images_comer()

        print("Getting Images from the divers")
        self.get_real_divers()

    def create_bounding_box(self, image_name, folder):
        label_name = image_name.replace("jpg", "txt")
        image = cv2.imread(f"{folder}/images/{image_name}")
        r = self.yolo.predict(image)

        max_confidence = 80
        x1, y1, x2, y2 = -1, -1, -1, -1
        for i, (x1, y1, x2, y2, confidence, _class) in enumerate(r.xyxy[0]):
            if confidence > max_confidence:
                x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
                max_confidence = confidence

        if x1 != -1:
            with open(f"{folder}/bounding_boxes/{label_name}", "w") as file:
                file.write(f"{x1} {y1} {x2} {y2}")

    def add_missing_bounding_boxes(self):
        images = os.listdir(self.real_folder + "/images")
        boxes = os.listdir(self.real_folder + "/bounding_boxes")

        for image in images:
            if image.replace("jpg", "txt") not in boxes:
                self.create_bounding_box(image, self.real_folder)
        
    def make_images_black_and_white(self):
        for image in os.listdir(f'{self.real_folder}/images/'):
            image_file = Image.open(f'{self.real_folder}/images/{image}') 
            image_file = image_file.convert('1') 
            image_file.save(f'{self.real_folder}/images/{image}')
            

    def create_test_real(self):

        image_path = self.real_folder + "/images"
        labels_path = self.real_folder + "/labels"

        images = os.listdir(image_path)
        images_test = sample(images, self.test_size)
        for image in images_test:
            label = image.replace("jpg", "txt")
            os.rename(
                f"{image_path}/{image}", f"{self.dataset_path}/test/images/{image}"
            )
            os.rename(
                f"{labels_path}/{label}", f"{self.dataset_path}/test/labels/{label}"
            )

    def create_train_real(self, size = None):
        path = self.real_folder + "/images"
        images = os.listdir(path)
        if size:
            images = sample(images, size)

        for image_name in tqdm(images, desc="Augmenting real"):
            new_image_name = f"{self.dataset_path}/train/images/{image_name}"
            label_name = image_name.replace("jpg", "txt")
            label_path = f"{self.dataset_path}/train/labels/"

            try:
                image = cv2.imread(f"{path}/{image_name}")
                
                with open(f"{self.real_folder}/labels/{label_name}") as f:
                    label = f.read()
                with open(f"{self.real_folder}/bounding_boxes/{label_name}") as f:
                    xyxy = f.read().split(" ")

                cropped = get_ROI(image, xyxy, 20)
                if cropped.shape[0] > cropped.shape[1]:
                    cropped = cv2.rotate(cropped, cv2.cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(new_image_name, cropped)
                with open(f"{label_path}/{label_name}", "w") as file:
                    file.write(label)
                Augmenter(
                    new_image_name, synth=False, 
                    label=label, 
                    label_path=label_path, 
                    scaling_factor = self.real_scaling_factor
                ).pipeline()

            except Exception as e:
                print(e)
                traceback.print_exc()

    def create_train_synth(self):
        path = self.synth_folder + "/images"
        images = os.listdir(path)

        for img_name in tqdm(images, desc="Augmenting synth"):
            image_name = img_name.replace("jpeg", "jpg")
            new_image_name = f"{self.dataset_path}/train/c/{image_name}"

            label_name = image_name.replace("jpg", "txt")
            label_path = f"{self.dataset_path}/train/labels/"

            image = cv2.imread(f"{path}/{img_name}")
            
            with open(f"{self.synth_folder}/labels/{label_name}") as f:
                label = f.read()
            
            with open(f"{label_path}/{label_name}", "w") as file:
                file.write(label)
            
            cv2.imwrite(new_image_name, image)
            Augmenter(
                new_image_name, 
                synth=True, 
                label=label, 
                label_path=label_path,
                scaling_factor = self.synth_scaling_factor
            ).pipeline()
            
            if uniform(0, 1) > self.prob_save_original_synth:
                os.remove(new_image_name)
                os.remove(new_image_name.replace('images','labels').replace('jpg','txt'))
