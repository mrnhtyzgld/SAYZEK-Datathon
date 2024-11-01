import json
import os

# JSON dosyasını yükleyin
with open('../data/annotations/train.json') as f:
    annotations = json.load(f)

# Her görüntü için bir .txt dosyası oluştur
for image_info in annotations['images']:
    image_id = image_info['id']
    image_name = image_info['file_name']
    annotations_for_image = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

    # .txt dosyasının yolunu belirleyin
    txt_file_path = os.path.join('../data/images/', image_name.replace('.png', '.txt'))

    with open(txt_file_path, 'w') as f_txt:
        for ann in annotations_for_image:
            class_id = ann['category_id']
            x, y, width, height = ann['bbox']
            # YOLO formatına dönüştürme
            x_center = x + width / 2
            y_center = y + height / 2
            f_txt.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

