import yolov5
from torch import cuda, nn
import cv2
import numpy as np
from utils import get_ROI


class Detector(nn.Module):
    def __init__(self, yolo_path, num_cameras, confidence=0.90):
        super().__init__()
        self._load_yoloModel(yolo_path)
        self.confidence = confidence
        self.num_cameras = num_cameras
        self.show = False

    # @timeit
    def forward(self, batch):
        results = self.Yolo.predict(batch)
        return self._handle_results(batch, results)

    def _handle_results(self, batch, results):
        filtered_batch = []
        good_frames = [[] for _ in range(self.num_cameras)]

        # iteramos imagenes
        for k, result in enumerate(results.xyxy):
            n_camera = k % self.num_cameras
            num_image = int(k / self.num_cameras)
            has_good_frame = False

            # faltan rotaciones y asi

            # iteramos detecciones por imagen
            for i, (x1, y1, x2, y2, confidence, _class) in enumerate(result):
                if results.names[int(_class)] != "barcode":
                    continue

                if confidence > self.confidence:
                    has_good_frame = True
                    cropped_img = get_ROI(batch[k], (x1, y1, x2, y2), margin_pct=20)
                    filtered_batch.append(cropped_img)

            if has_good_frame:
                good_frames[n_camera].append(1)
            else:
                good_frames[n_camera].append(0)

            if self.show:
                key = cv2.waitKey(1) & 0xFF
                cv2.imshow(f"Barcode reader {n_camera}", batch[k])

        good_frames = np.array(good_frames)
        good_frames = np.sum(good_frames, axis=0)

        return filtered_batch, good_frames.tolist()

    def _load_yoloModel(self, filepath):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.Yolo = yolov5.YOLOv5(filepath, device=self.device)
