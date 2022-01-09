from main import Camera


h,w = (1080, 1920)
fps = 60
batch_size = 5


try:
    C = Camera(fps, h, w, batch_size)
    C.test()
except:
    C.release()
    cv2.destroyAllWindows()