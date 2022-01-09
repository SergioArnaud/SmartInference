import cv2
import numpy as np
import random as rd
from random import uniform
import math
import os
from augmentation_utils import gamma_trans, random_box_size, rotate
from parameters_augmentations import parameters_real, parameters_synth, probs


class Augmenter:

    # Initialization
    def __init__(self, filename, synth=True, label=None, label_path="", scaling_factor = 1):
        self.path = os.path.dirname(filename)
        self.filename = filename.split("/")[-1].split(".")[0]
        self.original_image = cv2.imread(filename)
        self.image = self.original_image
        self.label = label
        self.label_path = label_path
        assert (
            label and label_path
        ), "If you send a label you should also send the path in which it is saved"
        
        self.probs = probs
        current_scaling_factor = sum([probs[category] for category in probs])
        prob_normalization_constant = scaling_factor / current_scaling_factor
        
        for category in self.probs:
            self.probs[category] = self.probs[category] * prob_normalization_constant
        
        if synth:
            self.parameters = parameters_synth
            self.heavy_noise('initial')
            self.original_image = self.image

        else:
            self.parameters = parameters_real

    # Rotation with white bound
    def random_rotation(self, d_range=[-10, 10]):
        """Rotates an image with a random angle

        Parameters
        ----------
        d_range : list, optional
            list with posible rangle of angles, by default [0, 360]

        Returns
        -------
        self
            Augmenter class
        """
        image = self.image
        angle = rd.randrange(d_range[0], d_range[1])
        self.image = rotate(image, angle)
        return self

    def upside_down(self):
        self.image = rotate(self.image, 180)
        return self

    # Guassian blur
    def blur(self):
        min_proportion = self.parameters["blur"]["min_proportion"]
        max_proportion = self.parameters["blur"]["max_proportion"]

        image = self.image

        # Get image shape
        h, w = image.shape[:2]

        # Blur proportion
        k = min(h, w)
        a = k * min_proportion / 2
        b = k * max_proportion / 2

        # Random kernel selection and ensure is an odd number
        s_kernel = np.random.randint(math.floor(a), math.ceil(b)) * 2 + 1

        # ksize
        ksize = (s_kernel, s_kernel)

        # Using cv2.blur() method
        self.image = cv2.blur(image, ksize)

        return self

    # Rotated and perspective transformed
    def rpt(self):
        image = self.image

        # Get image shape
        h, w = image.shape[:2]

        # Create src cube
        # top-left, top-right, bottom-right, bottom-left
        srcQuad = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Create random destiny cube within a range from -1.25 to 1.25 h and w
        dstQuad = np.float32(np.zeros((4, 2)))

        # Get modifications in extrema points
        ph = int(0.25 * h)
        pw = int(0.25 * w)

        # Top left
        a = rd.randrange(0, pw)
        b = rd.randrange(0, ph)
        dstQuad[0] = [0 + a, 0 + b]

        # Top right
        a = rd.randrange(0, pw)
        b = rd.randrange(0, ph)
        dstQuad[1] = [w - a, 0 + b]

        # Bottom right
        a = rd.randrange(0, pw)
        b = rd.randrange(0, ph)
        dstQuad[2] = [w - a, h - b]

        # Bottom Left
        a = rd.randrange(0, pw)
        b = rd.randrange(0, ph)
        dstQuad[3] = [0 + a, h - b]

        # Gettin perspective matrix
        p_matrix = cv2.getPerspectiveTransform(srcQuad, dstQuad)

        result = cv2.warpPerspective(
            image,
            p_matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        self.image = result
        return self

    # Cylindered and Curvy warped
    def ccw(self):
        image = self.image
        h, w = image.shape[:2]

        # Random focal lenght
        f = w + rd.randrange(-int(2 * w / 3), int(2 * w / 3))

        blank = np.zeros_like(image)

        # random center to change point of view
        center_y, center_x = int(h / 2), int(w / 2)

        center_y += rd.randrange(-int(h / 2), int(h / 2))
        center_x += rd.randrange(-int(w / 2), int(w / 2))

        # Apply tranformation for each point
        for y in range(h):
            for x in range(w):
                theta = math.atan((x - center_x) / f)
                point_x = int(f * math.tan((x - center_x) / f) + center_x)
                point_y = int((y - center_y) / math.cos(theta) + center_y)

                if point_x >= w or point_x < 0 or point_y >= h or point_y < 0:
                    blank[y, x, :] = 255
                else:
                    blank[y, x, :] = image[point_y, point_x, :]

        self.image = blank

        return self

    def motion_blur(self, parameters = 'motion_blur'):

        min_proportion = self.parameters[parameters]["min_proportion"]
        max_proportion = self.parameters[parameters]["max_proportion"]

        image = self.image

        k = min(image.shape[0], image.shape[1])
        a = k * min_proportion
        b = k * max_proportion
        kernel_size = max(np.random.randint(math.floor(a), math.ceil(b)), 2)

        orientation = rd.choice(["vertical", "horizontal", "diagonal"])
        if orientation == "vertical":
            kernel_size = int(kernel_size * 1.2)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        elif orientation == "horizontal":
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        elif orientation == "diagonal":
            kernel_size = int(kernel_size * .8)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel = np.identity(kernel_size)
        else:
            raise Exception("Orientation should be vertical or horizontal")

        kernel /= kernel_size
        self.image = cv2.filter2D(image, -1, kernel)
        return self

    def heavy_noise(self, parameter_name = 'heavy_noise'):
        mean_noise = self.parameters[parameter_name]["mean_noise"]
        variance_noice = self.parameters[parameter_name]["variance_noise"]

        # variance_noice = min(85, 10 + np.random.gamma(4.5, 4.2)
        image = self.image
        gaussian = np.random.normal(
            mean_noise, variance_noice, (image.shape[0], image.shape[1], 1)
        )
        noisy_image = image + gaussian
        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

        self.image = noisy_image.astype(np.uint8)
        return self

    def dark(self):
        min_gamma = self.parameters["dark"]["min_gamma"]
        max_gamma = self.parameters["dark"]["max_gamma"]

        self.image = gamma_trans(
            self.image, np.random.choice(np.linspace(min_gamma, max_gamma, 100))
        )
        return self

    def overexposed(self):
        min_gamma = self.parameters["overexposed"]["min_gamma"]
        max_gamma = self.parameters["overexposed"]["max_gamma"]

        self.image = gamma_trans(
            self.image, np.random.choice(np.linspace(min_gamma, max_gamma, 100))
        )
        return self

    def occluded(self, tilted=False):
        img = self.image.copy()

        # build oclussion Array [[[0,0,0], [1,1,1]]]
        h, w, ch = img.shape
        box_w, box_h = random_box_size(h, w)

        # Generate random coordinates
        x = np.random.randint(0, w - box_w)
        y = np.random.randint(0, h - box_h)
        if not tilted:
            img[y: y + box_h, x: x + box_w, :] = 0
            self.image = img
            return self
        else:
            box_container = np.ones_like(img)
            box_container[x: x + box_w, y: y + box_h, :] = 2

            angle = rd.randrange(0, 360)
            box_container = rotate(box_container, angle)[:h, :w, :]

            box_container = np.where(box_container < 2, 1, 0)
            self.image = (box_container * img).astype(np.uint8)
            return self

    def return_image(self):
        to_return = self.image
        self.image = self.original_image
        return to_return

    def save_image(self, name):
        
        with_prob = self.probs[name.replace(f'_{self.filename}','')]

        if uniform(0, 1) < with_prob:
            filaname = f"{self.path}/{name}.jpg"
            cv2.imwrite(filaname, self.image)

            if self.label:
                with open(f"{self.label_path}/{name}.txt", "w") as file:
                    file.write(self.label)
                    
        self.image = self.original_image

    def pipeline(self):

        self.dark().save_image(f"dark_{self.filename}")
        self.overexposed().save_image(f"overexposed_{self.filename}")

        self.occluded().save_image(f"occluded_{self.filename}")
        self.dark().occluded().save_image(
            f"dark_occluded_{self.filename}"
        )

        self.rpt().save_image(f"rtp_{self.filename}")
        self.dark().rpt().save_image(f"dark_rtp_{self.filename}")
        self.ccw().rpt().save_image(f"ccw_rpt_{self.filename}")

        self.ccw().save_image(f"ccw_{self.filename}")
        self.dark().rpt().save_image(f"dark_ccw_{self.filename}")

        self.occluded().rpt().save_image(f"ocluded_rtp_{self.filename}")

        self.blur().save_image(f"blur_{self.filename}")
        self.blur().rpt().save_image(f"rpt_blur_{self.filename}")
        self.blur().ccw().save_image(f"rpt_blur_{self.filename}")

        self.upside_down().blur().save_image(
            f"upside_down_blur_{self.filename}"
        )
        self.upside_down().dark().save_image(
            f"upside_down_dark_{self.filename}"
        )
        self.upside_down().ccw().save_image(
            f"upside_down_ccw_{self.filename}"
        )
        self.occluded().upside_down().save_image(
            f"upside_down_ocluded_{self.filename}"
        )

        self.heavy_noise().random_rotation().save_image(
            f"heavy_noise_rdmrot_{self.filename}"
        )
        self.overexposed().occluded().rpt().ccw().save_image(
            f"overexposed_occluded_rpt_ccw_{self.filename}"
        )
        self.dark().occluded().rpt().ccw().save_image(
            f"dark_occluded_rpt_ccw_{self.filename}"
        )
        self.occluded().rpt().ccw().save_image(
            f"occluded_rpt_ccw_{self.filename}"
        )

        self.occluded().overexposed().save_image(
            f"overexposed_occluded_{self.filename}"
        )
        self.heavy_noise().save_image(f"heavy_noise_{self.filename}")
        self.motion_blur().save_image(f"motion_blur_{self.filename}")

        self.blur().random_rotation().ccw().save_image(
            f"blur_ccw_rdmrot_{self.filename}"
        )
        self.dark().random_rotation().save_image(
            f"dark_rdmrot_{self.filename}"
        )
        self.blur().random_rotation().save_image(
            f"blur_rdmrot_{self.filename}"
        )
        self.heavy_noise().random_rotation().save_image(
            f"heavy_noise_rdmrot_{self.filename}"
        )
        self.motion_blur().random_rotation().save_image(
            f"mblur_rdmrot_{self.filename}"
        )

        self.motion_blur().ccw().rpt().save_image(
            f"ccw_rpt_mblur_{self.filename}"
        )
        self.overexposed().motion_blur('small_motion_blur').ccw().save_image(
            f"ccw_overexposed_mblur_{self.filename}"
        )
        self.dark().motion_blur('small_motion_blur').ccw().save_image(
            f"ccw_dark_mblur_{self.filename}"
        )
        self.dark().motion_blur('small_motion_blur').save_image(
            f"mblur_dark_{self.filename}"
        )
        self.overexposed().motion_blur('small_motion_blur').save_image(
            f"mblur_overexposed_{self.filename}"
        )
        self.dark().motion_blur('small_motion_blur').random_rotation().save_image(
            f"dark_rdmrot_mblur_{self.filename}"
        )
