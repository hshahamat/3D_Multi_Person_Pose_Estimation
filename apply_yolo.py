import numpy as np
import cv2
# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width=width, height=height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

def bbx4person(detections):
    per = []
    for d in detections:
        if d[0]=='person':
            per.append(d)
    return per
	
	
img = cv2.imread('tmp_img_2021.png')
# call our darknet helper on webcam image
detections, width_ratio, height_ratio = darknet_helper(img, width, height)

detections = bbx4person(detections)


bbox_list = []
for label, confidence, bbox in detections:
    left, top, right, bottom = bbox2points(bbox)
    ow = right-left
    oh = bottom-top
    bb=[int(left * width_ratio), int(top * height_ratio),
        int(ow * width_ratio), int(oh * height_ratio)]
    bbox_list.append(bb)

blist = np.array(bbox_list)
np.save('bbox.npy',blist)
#np.save('original_img.npy',img)
print('done ...')	
