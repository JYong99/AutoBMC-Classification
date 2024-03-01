import torch, evaluate, numpy as np, wandb, os, shutil, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    Compose,

    CenterCrop,
    Resize,
    Normalize,
    ToTensor,
)

start_time = time.time()

testset_size = 0.3

#Loading of dataset
ds = load_dataset("imagefolder", data_dir="/home/dxd_jy/joel/Capstone/For_Training/Training_Dataset")

#Mapping labels to numbers
labels1 = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels1):
    label2id[label] = i
    id2label[i] = label

#Load pre-trained model
model_checkpoint1 = "microsoft/beit-base-patch16-384"
image_processor1 = AutoImageProcessor.from_pretrained(model_checkpoint1)
model_path = "/home/dxd_jy/joel/Capstone/Model/beit-base-patch16-384-10L_10E_8B_5e-05_0.3"
weight1 = 0.9229

model_checkpoint2 = "google/vit-base-patch16-224"
image_processor2 = AutoImageProcessor.from_pretrained(model_checkpoint2)
model_path2 = "/home/dxd_jy/joel/Capstone/Model/best-vit-base-patch16-224-10L_20E_8B_5e-05_0.3"
weight2 = 0.9654

#Prepare Model 1
normalize = Normalize(mean=image_processor1.image_mean, std=image_processor1.image_std)
if "height" in image_processor1.size:
    size = (image_processor1.size["height"], image_processor1.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor1.size:
    size = image_processor1.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor1.size.get("longest_edge")

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

model = AutoModelForImageClassification.from_pretrained(
    model_path, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

#Prepare Model 2
normalize2 = Normalize(mean=image_processor2.image_mean, std=image_processor2.image_std)
if "height" in image_processor2.size:
    size2 = (image_processor2.size["height"], image_processor2.size["width"])
    crop_size2 = size2
    max_size2 = None
elif "shortest_edge" in image_processor2.size:
    size2 = image_processor2.size["shortest_edge"]
    crop_size2 = (size2, size2)
    max_size2 = image_processor2.size.get("longest_edge")

val_transforms2 = Compose(
        [
            Resize(size2),
            CenterCrop(crop_size2),
            ToTensor(),
            normalize2,
        ]
    )

def preprocess_val2(example_batch2):
    """Apply val_transforms across a batch."""
    example_batch2["pixel_values"] = [val_transforms2(image2.convert("RGB")) for image2 in example_batch2["image"]]
    return example_batch2

model2 = AutoModelForImageClassification.from_pretrained(
    model_path2, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

#Load metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""

    labels = eval_pred.label_ids
    predictions = np.argmax(eval_pred.predictions, axis=1)

    accuracy_results    = accuracy.compute(predictions=predictions, references=labels)
    f1_results          = f1.compute(predictions=predictions, references=labels, average="weighted")
    precision_results   = precision.compute(predictions=predictions, references=labels, average="weighted")
    recall_results      = recall.compute(predictions=predictions, references=labels, average="weighted")
    combined_results    = {**accuracy_results, **f1_results, **precision_results, **recall_results}
 
    return combined_results

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

name = "Test123"
args = TrainingArguments(
    name,
    remove_unused_columns=          False,
    evaluation_strategy=            "epoch",
    save_strategy=                  "epoch",
    gradient_accumulation_steps=    4,
    warmup_ratio=                   0.1,
    log_level=                      "info",
    logging_steps=                  1,
    logging_strategy=               "epoch",
    load_best_model_at_end=         True,
    save_total_limit=               1, 
    metric_for_best_model=          "accuracy",
)

#Split Dataset to get the eval_ds
seed = 1026847926404610400

splits = ds["train"].train_test_split(test_size=testset_size, stratify_by_column="label", seed=seed)
splits2 = splits['test'].train_test_split(test_size=0.5, stratify_by_column="label", seed=seed)
splits3 = splits['test'].train_test_split(test_size=0.5, stratify_by_column="label", seed=seed)

eval_ds = splits2['test']
eval_ds2 = splits3['test']
eval_ds.set_transform(preprocess_val)   
eval_ds2.set_transform(preprocess_val2)

#Evaluate Model 1
trainer1 = Trainer(
    model,
    args,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
final1 = trainer1.predict(test_dataset=eval_ds)

#Evaluate Model 2
trainer2 = Trainer(
    model2,
    args,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
final2 = trainer2.predict(test_dataset=eval_ds2)

#Remove empty folder
curr_dir = os.getcwd()
folder_name = name
folder_path = os.path.join(curr_dir, folder_name)

try:
    shutil.rmtree(folder_path)
except OSError as e:
    print(f"{e.strerror}")

#Do ensemble learning for both models (Soft Voting)(Weighted Average)
total = weight1 + weight2
ensemble_pred = []
for x in range(len(final1.predictions)):
    combined = ((weight1 * final1.predictions[x]) + (weight2 * final2.predictions[x]))/total
    ensemble_pred.append(np.argmax(combined))

wandb.init(project="Ensemble Evaluation", 
        name="Ensemble_Evaluation",
        config={
            "Model1": model_checkpoint1,
            "Model2": model_checkpoint2,
            "Total Evalutaion Data": f"Size: {0.5*(testset_size)}, Total: {len(eval_ds)}",
            "Pytorch GPU": torch.cuda.is_available(),
            "Seed": seed
        })

wandb.log({f"EnsemblePred_ConfMatrix" : wandb.plot.confusion_matrix( 
            preds=ensemble_pred, y_true=final1.label_ids,
            class_names=id2label
)})

accuracy_results = accuracy.compute(predictions=ensemble_pred, references=final1.label_ids)

end_time = time.time()
duration = (end_time - start_time)/60

wandb.log({"Test_Acc": accuracy_results,
           "Time_Taken(Mins)": duration
           })

wandb.finish(exit_code=0)