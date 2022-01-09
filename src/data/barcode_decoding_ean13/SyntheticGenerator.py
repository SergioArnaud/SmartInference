from barcode.writer import ImageWriter
from barcode.ean import EuropeanArticleNumber13 as EAN13
import os
import pandas as pd
import numpy as np


def _build_string(array):
    """Give an iterable and join it as a str.

    Args:array
    ([iterable): [array or list like, a collection of integers, to be joined
    as digits of the barcode]
    Returns:
        [str]: [barcode as str]
    """
    return "".join(array.astype(str))


def _read_label(path):
    with open(path, "r") as file:
        return next(file)


def _create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


class SyntheticGenerator:
    """Class for generating barcodes in a given path."""

    def __init__(self, 
                 path, 
                 n_barcodes, 
                 barcode_list=None, 
                 country_list=None,
                 manufacturer_list=None,
                 write_text=True):
        """Synthetic Generator Instance.

        Args:
            path ([str]): Name of the saving path,
                        if it does not exist it will be created
            n_barcodes ([int]): Number of barcodes to be generated
            barcode_list ([list], optional): List of barcode strs, to be generated
                                    first and in order. Defaults to None.
            write_text (bool, optional): Wether to write the human readable code at the
            bottom of the img. Defaults to True.
        """
        self.path = path
        self.write_text = write_text
        _create_folder(self.path)

        self.label_path = os.path.join(path, "labels")
        _create_folder(self.label_path)

        self.image_path = os.path.join(path, "images")
        _create_folder(self.image_path)

        self.n_barcodes = n_barcodes
        self.barcode_list = [] if barcode_list is None else barcode_list
        self.written = pd.Series()
        self.country_list = country_list
        self.manufacturer_list = manufacturer_list

    def generate(self):
        """Generate barcodes as indicated in init."""
        written_len = 0
        length = len(self.barcode_list)
        while written_len <= self.n_barcodes:
            if written_len >= length:
                code = ''
                if self.country_list:
                    code += str(np.random.choice(self.country_list))
                else:
                    code += _build_string(np.random.randint(0, 10, 3))
                if self.manufacturer_list:
                    code += str(np.random.choice(self.manufacturer_list))
                    
                remaining_numbers = 12 - len(code)
                code += _build_string(np.random.randint(0, 10, remaining_numbers))
                code += str(EAN13(code).calculate_checksum())
            else:
                code = self.barcode_list[written_len]

            code = EAN13(code, writer=ImageWriter(format="JPEG"))
            name = code.get_fullcode()
            if len(self.written[self.written == name]) == 0:
                self._save_image_barcode(code, written_len)
                written_len += 1

    def _save_image_barcode(self, code, written_len):
        name = code.get_fullcode()
        code.save(
            os.path.join(self.image_path, f"{written_len}"),
            options={"write_text": self.write_text},
        )
        with open(os.path.join(self.label_path, f"{written_len}.txt"), "w") as file:
            file.write(name)
        self.written.append(pd.Series([name]))
