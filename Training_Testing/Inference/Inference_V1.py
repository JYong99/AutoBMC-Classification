from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os, torch

model_checkpoint_test = "google/vit-base-patch16-224"
inf_image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_test)
model_test = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Model/best-vit-base-patch16-224-10L_20E_8B_5e-05_0.3")

def process_image(image):
    encoding = inf_image_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model_test(**encoding)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model_test.config.id2label[predicted_class_idx]

root_path = "/home/dxd_jy/joel/Capstone/Training_Testing/Test/Test_Images"
count = 0

def get_folder_names(directory):
    folder_names = []
    for entry in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, entry)):
            folder_names.append(entry)
    return folder_names

folders = get_folder_names(root_path)

for foldername in os.listdir(root_path):
    folder_path = os.path.join(root_path, foldername)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        test_img_buffer = Image.open(file_path)
        results = process_image(test_img_buffer)
        print(f"Label:{filename[:3]} -> Predicted:{results}")
        if filename[:3] == results:
            count = count + 1

print(f"\nTotal Correct Prediction: {count}")
print(f"Total Images: {len(os.listdir(folder_path))}")
