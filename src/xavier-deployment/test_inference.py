import os
import sys
import cv2

from orchestrator import Orchestrator
from detector import Detector
from camera import Camera

file_folder = os.path.dirname(__file__)
src_folder = '/'.join(file_folder.split('/')[:-1])
inference_folder = f'{src_folder}/inference'
sys.path.append(inference_folder)
from smart_inference.smart_inference import BarcodeInferencer

## Parameters
batch_size = 6
exploration_size = 20
top_k = 3
h,w = (1920, 1080)
fps = 30
yolo_confidence = .85
max_gap_continuity = 3
max_timegap = 1000
max_imgs_for_inference = 30
threshold_mdcnn = .9
show = True

print('-'*60)
print(' '*15, 'Setting up the environment')
print('-'*60)

C = Camera(fps, h, w, batch_size)
Y = Detector(f'{file_folder}/models/yolo_model.pt', C.num_cameras, yolo_confidence)
S = BarcodeInferencer(f'{file_folder}/models/inference_model.ckpt', exploration_size, top_k, threshold = threshold_mdcnn)
O = Orchestrator(C, Y, S, max_gap_continuity, max_timegap, max_imgs_for_inference, show)

print('-'*60)
print(' '*20, 'Ready to go!')
print('-'*60)
try:
    O.main_loop()
except Exception as e:
    print(e)
    C.release()
    cv2.destroyAllWindows()
