from time import time
import cv2


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print(method.__name__, (te - ts) * 1000)
        return result

    return timed


def get_ROI(image, xyxy, margin_pct):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))

    margin = int((x2 - x1) * margin_pct / 100)

    x1 = int(float(x1 - margin)) if x1 - margin >= 0 else int(float(x1))
    y1 = int(float(y1 - margin)) if y1 - margin >= 0 else int(float(y1))
    x2 = int(float(x2 + margin)) if x2 + margin <= image.shape[1] else int(float(x2))
    y2 = int(float(y2 + margin)) if y2 + margin <= image.shape[0] else int(float(y2))

    return image[y1:y2, x1:x2]

def put_text(frame, text, offset=0, fontScale=1.4):
    """Put a text on the upper left corner of the frame"""

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = fontScale
    fontColor = (0, 0, 255)
    lineType = cv2.LINE_AA
    upperLeftCornerOfText = (100, 80)

    cv2.putText(
        frame, str(text), upperLeftCornerOfText, font, fontScale, fontColor, lineType
    )

    return frame