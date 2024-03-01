from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os, torch

model_checkpoint_1 = "google/vit-base-patch16-224"
model_checkpoint_2 = "microsoft/beit-base-patch16-384"

inf_image_processor_1 = AutoImageProcessor.from_pretrained(model_checkpoint_1)
model_1 = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Model/best-vit-base-patch16-224-10L_20E_8B_5e-05_0.3")
inf_image_processor_2 = AutoImageProcessor.from_pretrained(model_checkpoint_2)
model_2 = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Training_Testing/Train/beit-base-patch16-384-10L_10E_8B_5e-05_0.3")

def process_image(image):

    results1 = {}
    results2 = {}
    combined_dict = {}
    index1= 0
    index2= 0

    model1weight = 0.9652
    model2weight = 0.9229
    total = model1weight + model2weight
    weight1 = model1weight/total
    weight2 = model2weight/total

    #Model1
    encoding = inf_image_processor_1(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model_1(**encoding)
        logits = outputs.logits
    for row in logits:
        for value in row:
            if value > 0:
                results1[model_1.config.id2label[index1]] = (value.item())
            index1 += 1
    sort_dict1 = dict(sorted(results1.items(), key=lambda item: item[1], reverse=True))

    #Model2
    encoding = inf_image_processor_2(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model_2(**encoding)
        logits = outputs.logits
    for row in logits:
        for value in row:
            if value > 0:
                results2[model_2.config.id2label[index2]] = value.item()
            index2 += 1
    sort_dict2 = dict(sorted(results2.items(), key=lambda item: item[1], reverse=True))
    
    # Update combined_dict with values from dict1
    for key, value in sort_dict1.items():
        if key in sort_dict2:
            combined_dict[key] = (value * weight1 + sort_dict2[key] * weight2) / 2
        else:
            combined_dict[key] = value * weight1
    # Update combined_dict with values from dict2 not already included
    for key, value in sort_dict2.items():
        if key not in combined_dict:
            combined_dict[key] = value * weight2

    # print(sort_dict1)
    # print(sort_dict2)       
    # print(combined_dict)
    sort_combined = dict(sorted(combined_dict.items(), key=lambda item: item[1], reverse=True))
    return next(iter(sort_combined))

root_path = "/home/dxd_jy/joel/Capstone/Dataset/Test"
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
