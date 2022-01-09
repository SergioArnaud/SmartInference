import cv2
import numpy as np
from utils import put_text

class Batch:
    def __init__(self, max_gap_continuity=1):
        self.frames = []
        self.unprocesed_frames = []
        self.good_frames = []
        self.barcode_frames = []
        self.timestamps = []
        self.is_continous = False
        self.has_barcodes = False
        self.time_elapsed = 0
        self.max_gap_continuity = max_gap_continuity

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        return str(
            {
                "frames": len(self.frames),
                "unprocesed_frames": len(self.unprocesed_frames),
                "good_frames": self.good_frames,
                "barcode_frames": len(self.barcode_frames),
                "timestamps": len(self.timestamps),
                "is_continous": self.is_continous,
                "has_barcodes": self.has_barcodes,
                "time_elapsed": self.time_elapsed,
            }
        )

    def add(self, frames, timestamps):
        self.frames.extend(frames)
        self.unprocesed_frames.extend(frames)
        self.timestamps.extend(timestamps)

    def add_good_frames(self, good_frames):
        self.good_frames.extend(good_frames)
        self.is_continous = self.__get_continuity()
        self.time_elapsed = max(self.timestamps) - min(self.timestamps)
        self.time_elapsed = self.time_elapsed.total_seconds() * 1000

    def add_barcode_frames(self, barcode_frames):
        if len(barcode_frames) > 0:
            self.barcode_frames.extend(barcode_frames)
            self.has_barcodes = True
        self.unprocesed_frames = []

    def __get_quality(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __get_continuity(self):
        if 1 in self.good_frames:
            first_ocurrence = self.good_frames.index(1)
        else:
            return False

        non_ocurrence_count = 0
        for value in self.good_frames[first_ocurrence:]:
            if not value:
                non_ocurrence_count += 1
            else:
                non_ocurrence_count = 0

            if non_ocurrence_count > self.max_gap_continuity:
                return False
        return True

    def reset(self):
        self.frames = []
        self.good_frames = []
        self.unprocesed_frames = []
        self.barcode_frames = []
        self.timestamps = []
        self.is_continous = False
        self.has_barcodes = False
        self.time_elapsed = 0

    def get_frames(self):
        return self.frames

    def get_unprocesed_frames(self):
        return self.unprocesed_frames

    def get_best_barcode_frames(self, proportion=0.5):
        scores = [self.__get_quality(frame) for frame in self.barcode_frames]
        score_cutoff = sorted(scores, reverse=True)[int(proportion * len(scores)) - 1]
        frames_ans = []

        best_frame, best_score = None, 0
        for score, frame in zip(scores, self.barcode_frames):
            if score >= score_cutoff:
                frames_ans.append(frame)
            if score > best_score:
                best_score = score
                best_frame = frame

        return frames_ans, best_frame

    def get_barcode_frames(self):
        return self.barcode_frames


class Orchestrator:
    def __init__(
        self,
        Camera,
        detector,
        barcode_inferencer,
        max_gap_continuity=1,
        max_timegap=500,
        max_imgs_for_inference=15,
        show=False,
    ):
        self.Camera = Camera
        self.detector = detector
        self.max_timegap = max_timegap
        self.barcode_inferencer = barcode_inferencer
        self.batch = Batch()
        self.batch_action = "New frame session"
        self.is_inference_ready = False
        self.max_imgs_for_inference = max_imgs_for_inference
        self.results = None
        self.show = show

        if show:
            self.Camera.set_windows()
            self.detector.show = True

    def main_loop(self):
        while True:
            frames, timestamps = self.Camera.get_batch()

            self.__handle_new_frames(frames, timestamps)

            self.__detect_barcodes()

            self.__set_next_action()

            # print(self.batch)

            if self.is_inference_ready:

                self.__send_for_inference()

                self.__process_result()

    def __handle_new_frames(self, frames, timestamps):
        if self.batch_action == "Continue frame session":
            self.batch.add(frames, timestamps)

        elif self.batch_action == "New frame session":
            self.batch.reset()
            self.batch.add(frames, timestamps)

    def __detect_barcodes(self):
        frames = self.batch.get_unprocesed_frames()
        detected_frames, good_frames = self.detector(frames)
        self.batch.add_good_frames(good_frames)
        self.batch.add_barcode_frames(detected_frames)

    def __set_next_action(self):
        self.is_inference_ready = False
        if (
            self.batch.is_continous
            and self.max_timegap > self.batch.time_elapsed
            and self.max_imgs_for_inference > len(self.batch.barcode_frames)
        ):
            self.batch_action = "Continue frame session"
        else:
            self.batch_action = "New frame session"
            if self.batch.has_barcodes:
                self.is_inference_ready = True

    def __send_for_inference(self):
        frames, best_frame = self.batch.get_best_barcode_frames()
        results = self.barcode_inferencer(frames)
        self.results = results
        self.best_frame = best_frame

    def __show_results(self):

        cv2.namedWindow("Barcode detected", cv2.WINDOW_NORMAL)
        cv2.imshow("Barcode detected", self.best_frame)

        if self.results:
            black_frame = np.zeros((128, 1024, 3))
            black_frame = put_text(black_frame, self.results[0][0], fontScale=3)
            cv2.namedWindow("Barcode number", cv2.WINDOW_NORMAL)
            cv2.imshow("Barcode number", black_frame)

    def __process_result(self):
        if self.results:
            if self.show:
                self.__show_results()

            print(self.results)
            self.results, self.best_frame = [], None
        else:
            print("No inference results")
