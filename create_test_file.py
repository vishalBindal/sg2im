import json
import os
import cv2

orig_size = (768, 1024)
image_size = (96, 128) 

def resize_bbox(bbox):
    x,y,w,h = bbox
    fx = image_size[1]/orig_size[1]
    fy = image_size[0]/orig_size[0]
    return [x*fx, y*fy, w*fx, h*fy]

img = {
    "objects": [],
    "relationships": [],
    "bboxes": []
}

val_id = 259

with open('datasets/instances_val.json') as f:
    val_data = json.load(f)
    for ann in val_data['annotations']:
        if ann['image_id'] == val_id:
            img['bboxes'].append(resize_bbox(ann['bbox']))
            cid = ann['category_id']
            img['objects'].append(val_data['categories'][cid]['name'])

images = [img]

with open('test_image.json', 'w') as f_out:
    json.dump(images , f_out)

# also copy relevant image
img_path = os.path.join('datasets/val_images', val_data['images'][val_id]['file_name'])
img = cv2.resize(cv2.imread(img_path), (image_size[1], image_size[0]))
cv2.imwrite('test.png', img)
