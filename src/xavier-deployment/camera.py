import cv2
from datetime import datetime


def test_device(n_camera):
    cap = cv2.VideoCapture(n_camera)
    if cap is None or not cap.isOpened():
        cap.release()
        return False
    return True


class Camera:
    def __init__(self, fps, width, height, batch_size) -> None:
        self.cameras = []
        self.fps = fps
        self.width = width
        self.height = height
        self.batch_size = batch_size

        self.release()
        self.__init_cameras()
        self.__set_fps()
        self.__set_buffer_size()

    def __init_cameras(self, max_cameras=4):
        for n_camera in range(max_cameras):
            if test_device(n_camera):
                camera = cv2.VideoCapture(n_camera)
                self.cameras.append(camera)
        self.num_cameras = len(self.cameras)

    def __set_fps(self):
        for camera in self.cameras:
            camera.set(cv2.CAP_PROP_FPS, self.fps)

    def __set_res(self):
        for camera in self.cameras:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def __set_buffer_size(self):
        for camera in self.cameras:
            camera.set(cv2.CAP_PROP_BUFFERSIZE, min(self.batch_size, 10))

    # @timeit
    def get_batch(self):
        self.frames, self.timestamps = [], []
        for _ in range(self.batch_size):
            for camera in self.cameras:
                self.__add_frame(camera)
        return self.frames, self.timestamps

    def __add_frame(self, camera):
        status, frame = camera.read()
        if status:
            self.frames.append(frame)
            self.timestamps.append(datetime.now())
        else:
            camera.release()

    def set_windows(self):
        for i in range(len(self.cameras)):
            cv2.namedWindow(f"Barcode reader {i}", cv2.WINDOW_NORMAL)

    def release(self):
        for camera in self.cameras:
            camera.release()

    def test(self):
        frames = []
        self.set_windows()

        while True:
            for i, frame in enumerate(frames):
                cv2.imshow(f"Barcode reader {i}", frame)

            frames = []

            for camera in self.cameras:
                ret, frame = camera.read()
                key = cv2.waitKey(1) & 0xFF
                if not ret:
                    return
                frames.append(frame)
