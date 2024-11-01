import json
import cv2
import os
from ultralytics import YOLO

# Path to test images and pre-trained YOLO model
# test_images_path = 'datathon/test/test_images' # test aşamasında yorumdan çıkar
test_images_path = 'test-images'
model_path = r"C:\Users\NihatEmreYüzügüldü\PycharmProjects\SAYZEK\runs\detect\train31\weights\best.pt"
model = YOLO(model_path)

# Load image file name to ID mapping
image_file_name_to_image_id = json.load(open('image_file_name_to_image_id.json'))

results = []
for img_name in os.listdir(test_images_path):
    image_path = os.path.join(test_images_path, img_name)

    # Perform inference directly
    inference_results = model(image_path)
    detections = inference_results[0].boxes  # Access bounding boxes, labels, and scores

    # Get image ID for the current image
    img_id = image_file_name_to_image_id[img_name]

    # Process each detection
    for det in detections:
        bbox = det.xyxy[0]  # Get bounding box in xyxy format
        bbox = bbox.clone()  # Create a normal tensor clone to avoid in-place modification restrictions
        bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]  # Convert xyxy to xywh
        label = int(det.cls) + 1  # Add 1 to label if your model output starts from 0
        score = det.conf

        # Prepare the result for this detection
        res = {
            'image_id': img_id,
            'category_id': label,
            'bbox': [float(coord) for coord in bbox.cpu().numpy()],  # Ensure each coord is a Python float
            'score': float(score.item())  # Ensure score is a Python float
        }

        results.append(res)

# Save results to a JSON file
with open('results-31.json', 'w') as f:
    json.dump(results, f)