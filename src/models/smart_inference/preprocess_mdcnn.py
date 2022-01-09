from torchvision import transforms
import math

proportions = (600,300)

def put_horizontal(img):
    if img.size[0] < img.size[1]:
        return img.rotate(90, expand = True)
    else:
        return img
    
def pad(img):
    w,h = img.size
    if w/h > proportions[0]/proportions[1]:
        value = w*proportions[1]/proportions[0]
        pad =  value - img.size[1]
        padding = (0,math.floor(pad/2),0,math.ceil(pad/2))
    else:
        value = h*proportions[0]/proportions[1]
        pad =  value - img.size[0]
        padding = (math.floor(pad/2),0,math.ceil(pad/2),0)
            
    return transforms.Pad(padding = padding, fill = 'white')(img)
        

preprocess_mdcnn = transforms.Compose(
            [
                transforms.Lambda(lambda img : put_horizontal(img)),
                transforms.Lambda(lambda img : pad(img)),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((proportions[1],proportions[0])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, .5, .5], 
                    std=[0.5, .5, .5]
                ),
            ]
            )

def get_ROI(image, xyxy, margin_pct):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))

    margin = int((x2 - x1) * margin_pct / 100)

    x1 = int(float(x1 - margin)) if x1 - margin >= 0 else int(float(x1))
    y1 = int(float(y1 - margin)) if y1 - margin >= 0 else int(float(y1))
    x2 = int(float(x2 + margin)) if x2 + margin <= image.shape[1] else int(float(x2))
    y2 = int(float(y2 + margin)) if y2 + margin <= image.shape[0] else int(float(y2))

    return image[y1:y2, x1:x2]
