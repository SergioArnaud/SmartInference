import os
import xml.etree.ElementTree as ET
import pandas as pd
import PIL.Image as Image
from tqdm import tqdm

def verify_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def parse_xml(path):
    """[Takes as input a path of an xml from the BDB Dataset and returns a list of dicts with the barcodes data]

    Args:
        path (str): A path to an XML from the DDB Dataset

    Returns:
        [list]: a list of observation dicts.
    """    '''
    '''
    real = True
    tree = ET.parse(path)
    polygons = tree.findall('.//Points')
    if len(polygons) == 0:
        polygons = tree.findall('.//Polygon')
        if len(polygons) > 0:
            real = False
    n_codes = len(polygons)
    data = []
    for polygon in polygons:
        points = pd.DataFrame([{'x':float(point.attrib['X']), 'y':float(point.attrib['Y'])}  for point in polygon])
        #points = [points[0], points[2]]
        if real:
            value = tree.find('.//Value').text
        else:
            value = 0
        barcode_type = tree.find('.//Barcode').attrib['Type']
        data.append({
            'label':['barcode'],
            'points': points,
            'value': value,
            'barcode_type': barcode_type
        })
    return data


def build_img_dict(path):
    """Builds a dict containing the information form an image of the BDB dataset

    Args:
        path (str): A path to an XML from the DDB Dataset

    Returns:
        [dict]: a dict with the information of an image
    """    
    img_path = path.replace('Markup', 'Image').replace('xml', 'png')

    data = {
        'annotation': parse_xml(path),
        'content': img_path
    }
    return data


def BDB2decoding_dataset(markup_path):
    """[Given the path of a ZVZ Real Dataset, builds a dataset for decoding EAN13 Barcodes] 

    Args:
        markup_path (str-path): [Path to the Markup folder inside a ZVZ-real Dataset]
    """    
    xmls = os.listdir(markup_path)
    real_barcodes = [build_img_dict(base_path + '/' + xml) for xml in xmls]
    ean13 = [(i['content'].replace('.png', ''), i['annotation'][0]['value']) for i in real_barcodes if i['annotation'][0]['barcode_type'] == 'EAN13' and len( i['annotation'][0]['value']) == 13]
    base_image_path = markup_path.replace('Markup', 'Image')
    image_paths = os.listdir(base_image_path)
    type_less_image_paths = [i.split('.')[0] for i in image_paths]


    verify_dir('Real_EAN13')
    verify_dir(os.path.join('Real_EAN13', 'Image'))
    verify_dir(os.path.join('Real_EAN13', 'Label'))

    for i, pair in enumerate(tqdm(ean13)):
        path, code = pair
        looking_path = path.split('/')[1]
        index = type_less_image_paths.index(looking_path)
        img_path = image_paths[index]
        img = Image.open(os.path.join( base_image_path,img_path)).convert("RGB")
        img.save(os.path.join('Real_EAN13', 'Image', str(i)+'.jpg'), "JPEG")

        label_name = f"{i}.txt"

        with open((os.path.join('Real_EAN13', 'Label', label_name)),"w") as label_file:
            label_file.write(code)
            
if __name__ == "__main__":
    markup_path = os.path.join('ZVZ-real-512', 'Markup')
    BDB2decoding_dataset(markup_path)
        
    