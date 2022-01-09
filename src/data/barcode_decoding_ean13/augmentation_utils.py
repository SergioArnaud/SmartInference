import numpy as np
import cv2


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_box_size(img_h, img_w):
    img_a = img_h * img_w
    box_a = np.random.randint(img_a / 12, img_a / 8)
    sqr = int(np.sqrt(box_a))
    box_h = np.random.randint(sqr / 2, sqr)
    box_w = box_a // box_h
    if box_w > img_w / 2:
        box_w = img_w // 2
    if box_h > img_h / 2:
        box_h = img_h // 2
    return box_w, box_h


def rotate(image, angle):
    # Get image center
    h, w = image.shape[0:2]
    centreY, centreX = h // 2, w // 2

    # Get matrix
    rot_mat = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)

    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = rot_mat[0][0]
    sinofRotationMatrix = rot_mat[1][0]

    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    newImageWidth = int(
        (h * np.abs(sinofRotationMatrix)) + (w * np.abs(cosofRotationMatrix))
    )
    newImageHeight = int(
        (h * np.abs(cosofRotationMatrix)) + (w * np.abs(sinofRotationMatrix))
    )

    # Get image center transformed
    v = [centreX, centreY, 1]
    centre = np.dot(rot_mat, v)

    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rot_mat[0][2] += (newImageWidth / 2) - centre[0]
    rot_mat[1][2] += (newImageHeight / 2) - centre[1]

    # Apply transformation with white bound
    result = cv2.warpAffine(
        image,
        rot_mat,
        (newImageWidth, newImageHeight),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return result