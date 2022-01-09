import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import git
import yaml
import shutil
import xml.etree.ElementTree as ET
import pandas as pd
import random
import zipfile38 as zipfile

repo = git.Repo(".", search_parent_directories=True)
base_path = f"{repo.working_tree_dir}/src"
os.chdir(base_path)

from utils.file_handling import (  # noqa : E402
    download_s3_file,
    unzip,
    upload_to_s3,
    object_exists,
)

def iterate_members(zip_file_like_object):
    zflo = zip_file_like_object
    assert zipfile.is_zipfile(zflo) # Here is where the error happens.
    # If I comment out the assert, the same error gets thrown on this next line:
    with zipfile.ZipFile(zflo) as zip:
        members = zip.namelist()
        for member in members:
            yield member

#fn = "filename.zip"
#iterate_members(open(fn, 'rb'))


def parse_xml(path):
    """
    This functions recieves a path of an xml file and returns a dictionary
    of the parsed information
    """
    real = True
    tree = ET.parse(path)
    polygons = tree.findall(".//Points")
    if len(polygons) == 0:
        polygons = tree.findall(".//Polygon")
        if len(polygons) > 0:
            real = False
    data = []
    for polygon in polygons:
        points = pd.DataFrame(
            [
                {"x": float(point.attrib["X"]), "y": float(point.attrib["Y"])}
                for point in polygon
            ]
        )
        if real:
            value = tree.find(".//Value").text
        else:
            value = 0
        barcode_type = tree.find(".//Barcode").attrib["Type"]
        data.append(
            {
                "label": ["barcode"],
                "points": points,
                "value": value,
                "barcode_type": barcode_type,
            }
        )
    return data


class CocoBarcodeDetectionBenchmark:
    """
    THis class:
        1. Downloads barcode dataset .zip files from s3
        2. Processes the barcode datasets so that they're in coco format
        3. Upload coco-formatted datasets to s3 (so that in the future) they can
            only be downloaded
        4. writes the coco yaml

    """

    def __init__(self, base_path, experiment, save_in_s3 = False, num_synth = 30000, num_negative = 1000, num_aug_synth = 30000, test_size = 0.1, small = True, create_files = True, filter_barcodes = []):
        self.base_path = base_path
        self.barcode_path = f"{base_path}/data/barcode_detection_benchmark"
        self.path = f"{base_path}/data/barcode_detection_benchmark/{experiment}"
        self.images_path = f"{self.path}/images"
        self.labels_path = f"{self.path}/labels"
        self.yaml_path = f"{self.path}/barcodes.yaml"
        self.s3_bucket = "la-comer"
        self.experiment = experiment
        self.num_synth = num_synth
        self.num_negative = num_negative
        self.num_aug_synth = num_aug_synth
        self.small = small
        self.test_size = test_size 
        self.filter_barcodes = filter_barcodes
        
        os.makedirs(self.path, exist_ok = True)

        if not os.path.exists(self.images_path):
            print("Creating coco dataset")
            self.__create_coco_dataset()
            
        if create_files:
            self._create_files(save_in_s3)


    def __get_s3_path(self, path, with_experiment=True):
        """Given a local path this function returns the correspondant s3_path

        Parameters
        ----------
        path : str
            Local path

        Returns
        -------
        str
            s3 path
        """
        if with_experiment:
            return path.replace(self.base_path + "/", "")
        return path.replace(self.base_path + "/", "").replace(self.experiment+'/','')

    
    def __filter_by_barcode_type(self, barcodes):
        if not self.filter_barcodes:
            return barcodes
        
        filtered = []
        for b in barcodes:

            try:
                if b['barcode_type'] in self.filter_barcodes:
                    filtered.append(b)
            except:
                if len(b['annotation']) == 0:
                    filtered.append(b) 
                    
                elif b['annotation'][0]['barcode_type'] in self.filter_barcodes:
                    filtered.append(b)            
                
        return filtered
                    
    
    def __create_coco_dataset(self):
        """Function that processes and creates coco dataset 
        """

        s3_images = self.__get_s3_path(self.images_path)
        s3_labels = self.__get_s3_path(self.labels_path)

        # If it's already in s3 or not
        if object_exists(self.s3_bucket, s3_images):
            print("Already in s3, no further processing, just downloading")
            download_s3_file(self.s3_bucket, s3_images, self.images_path)
            download_s3_file(self.s3_bucket, s3_labels, self.labels_path)
            return

        print("Coco Dataset not in s3, downloading .zip")
        self.__download_unprocessed_datasets()
        
        print('xml -> dicts')
        barcodes = self.__get_real_barcodes()
        
        if self.num_synth > 0:
            syn = self.__get_synth_barcodes()
            syn = self.__filter_by_barcode_type(syn)
            if self.num_synth > len(syn):
                self.num_synth = len(syn)
            barcodes += random.sample(syn, self.num_synth)
            
        barcodes += self.__get_augm_barcodes()
        
        if self.num_negative > 0:
            barcodes += self.__get_negatives()
            
        if self.num_aug_synth > 0:
            syn_aum = self.__get_synth_barcodes(augmented = True)
            syn_aum = self.__filter_by_barcode_type(syn_aum)
            if self.num_aug_synth > len(syn_aum):
                self.num_aug_synth = len(syn_aum)
            barcodes += random.sample(syn_aum, self.num_aug_synth)
        
        self.barcodes = self.__filter_by_barcode_type(barcodes)
    
    def _create_files(self, save_in_s3):
        print("Create train-test yolo files")
        train_codes, val_codes = train_test_split(self.barcodes, test_size=self.test_size)
        self.__create_yolo_dataset(train_codes, "train")
        self.__create_yolo_dataset(val_codes, "val")
        
        if save_in_s3:
            # upload s3
            s3_images = self.__get_s3_path(self.images_path)
            upload_to_s3(self.s3_bucket, self.images_path, s3_images)

            s3_labels = self.__get_s3_path(self.labels_path)
            upload_to_s3(self.s3_bucket, self.labels_path, s3_labels)

        self.__write_yaml()


    def __get_real_barcodes(self):
        """Function that processes the xml files with real barcodes

        Returns
        -------
        list
            List of dictionaries with the real barcode features
        """
        path = f"{self.path}/barcode_real_dataset/ZVZ-real/Markup"
        xmls = [f"{path}/{xml}" for xml in os.listdir(path)]
        return [
            self.__build_img_dict(xml)
            for xml in tqdm(xmls, position=0)
            if ".xml" in xml
        ]

    def __get_synth_barcodes(self, augmented = False):
        """Function that processes the xml files with synth barcodes

        Returns
        -------
        list
            List of dictionaries with the synth barcode features
        """
        if not augmented:
            path = f"{self.path}/barcode_synth_dataset/ZVZ-synth-512"
        else:
            if self.small:
                path = f"{self.path}/small_barcode_syn_aug/small_barcode_syn_aug"
            else:
                path = f"{self.path}/barcode_syn_aug/barcode_syn_aug"
            
        barcodes = []
        for folder in os.listdir(path):
            if "part" in folder:
                current_path = f"{path}/{folder}/Markup"
                xmls = [f"{current_path}/{xml}" for xml in os.listdir(current_path)]
                barcodes += [
                    self.__build_img_dict(xml)
                    for xml in tqdm(xmls, position=0)
                    if ".xml" in xml
                ]

        return barcodes
    
    def __get_augm_barcodes(self):
        path = f'{self.path}/barcode_augmented_dataset/barcode_augmented_dataset/Markup'
        xmls = [f'{path}/{xml}' for xml in os.listdir(path)]
        return [self.__build_img_dict(xml) for xml in tqdm(xmls, position=0) if '.xml' in xml]
    
    def __get_negatives(self):
        path = f'{self.path}/negative_images'
        image_paths = [path + '/' + i for i in os.listdir(path)][:self.num_negative]
        list_of_dicts = [{'annotation':[], 'content':image_path} for image_path in image_paths]
        return list_of_dicts
            
    
    def __build_img_dict(self, path):
        """Function that builds the dictionarie for an xml file

        Parameters
        ----------
        path : str
            Path of xml file

        Returns
        -------
        dict
            Dictionary  with the image and the markup
        """

        image_path = "/".join(path.replace("Markup", "Image").split("/")[:-1])
        filename = path.split("/")[-1].replace(".xml", "").replace("[", "[[]")
        image_path = next(Path(image_path).glob(f"{filename}*"))

        data = {"annotation": parse_xml(path), "content": image_path}
        return data

    def __create_yolo_dataset(self, barcodes, dataset_type):
        """Function that creates the yolo dataset

        Parameters
        ----------
        barcodes : List
            List with the barcode dicts
        dataset_type : str
            validation, training or test
        """

        images_path = Path(f"{self.images_path}/{dataset_type}")
        images_path.mkdir(parents=True, exist_ok=True)

        labels_path = Path(f"{self.labels_path}/{dataset_type}")
        labels_path.mkdir(parents=True, exist_ok=True)

        for img_id, row in enumerate(tqdm(barcodes, position=0)):

            image_name = f"{img_id}.jpeg"
            img = Image.open(row["content"]).convert("RGB")
            w, h = img.size
            img.save(f"{images_path}/{image_name}", "JPEG")

            label_name = f"{img_id}.txt"
            with open(f"{labels_path}/{label_name}", "w") as label_file:

                for a in row["annotation"]:

                    for label in a["label"]:

                        category_idx = 0

                        points = a["points"]
                        diffs = points.max() - points.min()
                        means = points.mean()
                        x = means.loc["x"] / w
                        y = means.loc["y"] / h

                        label_file.write(
                            f"{category_idx} {x} {y} {diffs.loc['x']/w} {diffs.loc['y']/h}\n"
                        )

                        
    def __download_unprocessed_datasets(self):

        # Download real dataset
        path_real = f'{self.path}/barcode_real_dataset.zip'
        if not os.path.exists(path_real.replace('.zip','')):
            s3_real = self.__get_s3_path(path_real, with_experiment = False)
            download_s3_file(self.s3_bucket, s3_real, path_real)
            path_real = unzip(path_real)
            print('unzipped')
        # Download synthetic create data
        if self.num_synth > 0:
            path_synth = f'{self.path}/barcode_synth_dataset.zip'
            if not os.path.exists(path_synth.replace('.zip','')):
                s3_synth = self.__get_s3_path(path_synth, with_experiment = False)
                download_s3_file(self.s3_bucket, s3_synth, path_synth)
                path_synth = unzip(path_synth)
                print('unzipped')
        # Download augmented data (rotations, noise, blur) based on real data
        path_augm = f'{self.path}/barcode_augmented_dataset.zip'
        if not os.path.exists(path_augm.replace('.zip','')):
            s3_augm = self.__get_s3_path(path_augm, with_experiment = False)
            download_s3_file(self.s3_bucket, s3_augm, path_augm)
            path_augm = unzip(path_augm)
            print('unzipped')
        
        if self.num_negative > 0:
            #download negative data
            path_negative = f'{self.path}/negative_images.zip'
            if not os.path.exists(path_negative.replace('.zip','')):
                s3_negative = 'data/barcode_detection_benchmark/negative_images.zip'
                download_s3_file(self.s3_bucket, s3_negative, path_negative)
                path_negative = unzip(path_negative)
                print('unzipped')
        
        if self.num_aug_synth > 0:
            if self.small:
                #download augmented synthetic data
                path_syn_aug = f'{self.path}/small_barcode_syn_aug.zip'
                if not os.path.exists(path_syn_aug.replace('.zip','')):
                    s3_syn_aug = 'data/barcode_detection_benchmark/small_barcode_syn_aug.zip'
                    download_s3_file(self.s3_bucket, s3_syn_aug, path_syn_aug)
                    path_syn_aug = unzip(path_syn_aug)
                    print('unzipped')
            else:
                #download augmented synthetic data
                path_syn_aug = f'{self.path}/barcode_syn_aug.zip'
                if not os.path.exists(path_syn_aug.replace('.zip','')):
                    s3_syn_aug = 'data/barcode_detection_benchmark/barcode_syn_aug.zip'
                    download_s3_file(self.s3_bucket, s3_syn_aug, path_syn_aug)
                    iterate_members(open(path_syn_aug, 'rb'))
                    print('unzipped')
        
    def __write_yaml(self):
        """Writes the coco yaml
        """
        data = dict(
            train=f"{self.images_path}/train",
            val=f"{self.images_path}/val",
            nc=1,
            names=["barcode"],
        )

        with open(self.yaml_path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)


if __name__ == "__main__":
    CocoBarcodeDetectionBenchmark()