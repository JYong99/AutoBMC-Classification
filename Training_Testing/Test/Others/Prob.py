from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
import torch
from PIL import Image

#google/vit-base-patch16-224
model_checkpoint_test = "microsoft/beit-base-patch16-384"
inf_image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_test)
model_test = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Training_Testing/Train/beit-base-patch16-384-10L_10E_8B_5e-05_0.3")

image = Image.open("/home/dxd_jy/joel/Capstone/Training_Testing/Test/Test_Images/Images/NGB_09966.jpg")

encoding = inf_image_processor(image.convert("RGB"), return_tensors="pt")

results = {}
index = 0
# forward pass
with torch.no_grad():
    outputs = model_test(**encoding)
    logits = outputs.logits
for row in logits:
    for value in row:
        if value > 0:
            results[model_test.config.id2label[index]] = value.item()
        index += 1

sort_dict = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
print(sort_dict) #Confidence Score
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model_test.config.id2label[predicted_class_idx])
