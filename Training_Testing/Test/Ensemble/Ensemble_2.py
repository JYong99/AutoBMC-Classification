from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image
from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="/home/dxd_jy/joel/Capstone/Training_Testing/Test/Test_Images/Test2/EOS")['train']

model_checkpoint_1 = "google/vit-base-patch16-224"
model_checkpoint_2 = "microsoft/beit-base-patch16-384"

inf_image_processor_1 = AutoImageProcessor.from_pretrained(model_checkpoint_1)
model_1 = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Model/best-vit-base-patch16-224-10L_20E_8B_5e-05_0.3")
pipe1 = pipeline("image-classification", model_1, image_processor=inf_image_processor_1)

inf_image_processor_2 = AutoImageProcessor.from_pretrained(model_checkpoint_2)
model_2 = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Training_Testing/Train/beit-base-patch16-384-10L_10E_8B_5e-05_0.3")
pipe2 = pipeline("image-classification", model_2, image_processor=inf_image_processor_2)

#image = Image.open("/home/dxd_jy/joel/Capstone/Training_Testing/Test/Test_Images/Images/NGB_09966.jpg")

weight1 = 0.9653
weight2 = 0.9229
total = weight1 + weight2
weight1 = weight1/total
weight2 = weight2/total

batch_size = 128

results = []
for i in range(0, len(ds), batch_size):
    print(i)
    batch = ds[i:i+batch_size]
    for image in batch["image"]:
        combined_results = {}
        result1 = pipe1(image)
        result2 = pipe2(image)

        for item in result1:
            combined_results[item['label']] = item['score'] * weight1
        for item in result2:
            if item['label'] in combined_results:
                combined_results[item['label']] += item['score'] * weight2
            else:
                combined_results[item['label']] = item['score'] * weight2
        #print(combined_results)

# results1 = pipe1(ds)
# results2 = pipe2(ds)

# combined_results = {}

# for item in results1:
#     combined_results[item['label']] = item['score'] * weight1

# for item in results2:
#     if item['label'] in combined_results:
#         combined_results[item['label']] += item['score'] * weight2
#     else:
#         combined_results[item['label']] = item['score'] * weight2

# print(combined_results)
