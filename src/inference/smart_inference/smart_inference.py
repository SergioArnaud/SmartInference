from PIL import Image
import torch
import os
import git
import cv2
from torch.nn.functional import softmax, normalize, threshold
from torch import nn, cuda
import pandas as pd

repo = git.Repo(".", search_parent_directories=True)
base_path = f"{repo.working_tree_dir}/src"

import sys

sys.path.append(base_path + "/models" + "/smart_inference")
from MDCNN_trainer_pl import Lit_MDCNN
from preprocess_mdcnn import preprocess_mdcnn



def checksum(arr):
    L = arr.shape[0]
    value, constant = 0, 1
    for i in range(1, L + 1):
        value += arr[L - i] * constant
        constant = 1 if constant == 3 else 3

    return value % 10 == 0


def cv2_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


class BarcodeInferencer(nn.Module):
    def __init__(self, mdcnn_path: str, size_exploration=25, topk=5, threshold = .8):
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = Lit_MDCNN.load_from_checkpoint(mdcnn_path)
        self.model.eval()
        self.threshold = threshold
        self.size_exploration = size_exploration
        self.topk = topk

        if self.device == "cuda":
            self.model = self.model.to("cuda")
            self.model.eval()

    def forward(self, frames):
        if not frames:
            return []
        self.clean()

        self.frames = frames
        self.preprocess_frames()

        if self.device == "cuda":
            self.frames = self.frames.to("cuda")

        out = self.model(self.frames)
        out = torch.stack(out)

        indexes = self.get_index_lists(out)
        self.result_smart_inference = []

        for index in indexes:
            self.options = {}
            out_subset = out[:, index, :]

            self.calculate_joint_probabilities(out_subset)
            self.initial_option = self.sorted_probabilities[:, 0]

            if checksum(self.initial_option) and self.topk == 1:
                self.result_smart_inference.append([self.get_barcode_str(self.initial_option)])
            else:
                posibilities = self.get_digit_posibilities()
                self.log_checksum(self.initial_option)
                self.explore(self.initial_option, posibilities)

                if self.options:
                    self.result_smart_inference.append(
                        sorted(
                            self.options, key=lambda x: self.options[x], reverse=True
                        )[0 : self.topk]
                    )
        
        return self.result_smart_inference

    def preprocess_frames(self):
        self.frames = [cv2_to_pil(img) for img in self.frames]
        self.frames = [preprocess_mdcnn(frame) for frame in self.frames]
        self.frames = torch.stack(self.frames)

    def get_barcode_str(self, tensor):
        return "".join([str(i) for i in tensor.tolist()])

    def split_out(self, out):
        n_frames = out.shape[1]
        out = [out[:, i, :].reshape(1, -1).detach() for i in range(n_frames)]
        out = torch.stack(out).view(n_frames, -1)
        return out, n_frames

    def similarity_matrix(self, tensor):
        tensor =  tensor / tensor.norm(dim=1)[:, None]
        return tensor @ tensor.T

    def get_index_lists(self, out):
        splitted, n_frames = self.split_out(out)
        sim_matrix = self.similarity_matrix(splitted).cpu().numpy()

        list_of_names = []

        def sim_filter(x, list_of_names, threshold):
            x = x[x > threshold]
            list_of_tuples = [(i, x.name) for i in x.index if x.name != i]
            if len(list_of_tuples) > 0:
                list_of_names.append(list_of_tuples)

        # function to apply to get list of list of tuples
        pd.DataFrame(sim_matrix).apply(
            lambda x: sim_filter(x, list_of_names, threshold=self.threshold)
        )
        # array = None
        # freeing more memory
        existing_dict = {i: None for i in range(n_frames)}
        for list_of_tuples in list_of_names:
            for i, j in list_of_tuples:
                if existing_dict[i] == None and existing_dict[j] == None:
                    existing_dict[i], existing_dict[j] = i, i
                elif existing_dict[i] == None and not existing_dict[j] == None:
                    existing_dict[i] = existing_dict[j]
                elif not existing_dict[i] == None and existing_dict[j] == None:
                    existing_dict[j] = existing_dict[i]
                else:
                    continue

        existing_dict = {
            value: item for value, item in existing_dict.items()
        }  # if item != None}
        fathers = []
        for key, value in existing_dict.items():
            if key == value or value is None:
                fathers.append([key])

        for key, value in existing_dict.items():
            if key == value:
                continue
            for i in fathers:
                if value == i[0]:
                    i.append(key)
        return fathers

    def calculate_joint_probabilities(self, out_subset):
        out = out_subset
        probabilities = softmax(out, 2)
        self.pre_prod = probabilities
        probabilities = torch.prod(probabilities, 1)
        self.post_prod = probabilities
        self.probabilities = normalize(probabilities, dim=1).cpu()
        self.sorted_probabilities = torch.argsort(
            self.probabilities, descending=True
        ).cpu()
        self.frames = self.frames.cpu()

    def get_top_gaps(self):
        gaps, num = {}, self.size_exploration

        for position, digit_probabilities in enumerate(self.probabilities):
            sorted_vals = self.sorted_probabilities[position]
            p0 = digit_probabilities[sorted_vals[0]]

            for digit_1 in sorted_vals:
                p1 = digit_probabilities[digit_1]
                gaps[f"{position}-{digit_1}"] = float(1 - (p0 - p1) / p0)

        top_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:num]
        return top_gaps

    def get_digit_posibilities(self):

        posibilities, top_gaps = {}, self.get_top_gaps()
        for gap, gain in top_gaps:
            position, digit = gap.split("-")
            if position not in posibilities:
                posibilities[position] = []
            posibilities[position].append(float(digit))
        posibilities = {
            key: value[1:] for key, value in posibilities.items() if len(value) > 1
        }
        return posibilities

    def log_checksum(self, combination):
        if checksum(combination):
            barcode = self.get_barcode_str(combination)
            probability = torch.prod(
                self.probabilities.gather(1, combination.reshape(13, 1))
            )
            self.options[barcode] = float(probability)

    def explore(self, combination, posibilities, combination_position=0):

        for position, value in enumerate(combination[combination_position:]):
            position = combination_position + position
            if not str(position) in posibilities:
                continue

            new_combination = torch.clone(combination)
            for posibility in posibilities[str(position)]:
                new_combination[position] = float(posibility)
                self.log_checksum(new_combination)
                self.explore(new_combination, posibilities, position + 1)

    def clean(self):
        self.sorted_probabilities = None
        self.frames = None
        self.probabilities = None
        self.result_smart_inference = None
        self.initial_option = {}
        self.options = {}
