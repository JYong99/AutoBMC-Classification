import torch, evaluate, numpy as np, wandb, random, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    Compose,
    
    GaussianBlur,
    ColorJitter,
    RandomAffine,
    RandomVerticalFlip,
    RandomRotation,
    RandomHorizontalFlip,
    RandomResizedCrop,

    CenterCrop,
    Resize,
    Normalize,
    ToTensor,
)
# seed = torch.random.initial_seed()
# torch.manual_seed(seed)

#Parameters
testset_size = 0.2
epoch = 5
batch_size = 4
lr = 5e-5
ds = load_dataset("imagefolder", data_dir="/home/dxd_jy/joel/Capstone/For_Training/Training_Dataset")

labels1 = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels1):
    label2id[label] = i
    id2label[i] = label

name = f"{len(id2label)}L_{epoch}E_{batch_size}B_{lr}_{testset_size}"

models = [
    "google/mobilenet_v2_1.4_224",
]

# duration = 20 * 60
# time.sleep(duration)

for x in range(len(models)):
    model_checkpoint = models[x]
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
            [
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                RandomResizedCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

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
    
        wandb.log({"Train_f1": f1_results["f1"], "Train_prec": precision_results["precision"], "Train_recall": recall_results["recall"]})
        return combined_results

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    model_name = model_checkpoint.split("/")[-1]
    run_name = f"{model_name}-{name}"

    args = TrainingArguments(
        run_name,
        remove_unused_columns=          False,
        evaluation_strategy=            "epoch",
        save_strategy=                  "epoch",
        learning_rate=                  lr,
        per_device_train_batch_size=    batch_size,
        gradient_accumulation_steps=    4,
        per_device_eval_batch_size=     batch_size,
        num_train_epochs=               epoch,
        warmup_ratio=                   0.1,
        log_level=                      "info",
        logging_steps=                  1,
        logging_strategy=               "epoch",
        load_best_model_at_end=         True,
        save_total_limit=               1, 
        metric_for_best_model=          "accuracy",
        report_to=                      "wandb"
    )

    seed = 1026847926404610400

    # split up training into training + validation and evaluation
    splits = ds["train"].train_test_split(test_size=testset_size, stratify_by_column="label", seed=seed)
    train_ds = splits['train']
    val_ds = splits['test']

    splits2 = val_ds.train_test_split(test_size=0.5, stratify_by_column="label", seed=seed)
    val_ds = splits2['train']
    eval_ds = splits2['test']

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val) 
    eval_ds.set_transform(preprocess_val)   

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    wandb.init(project="Training", 
            name=run_name,
            config={
                "Model": model_name,
                "Version": name,
                "Labels": id2label,
                "Epochs": epoch,
                "Batch_size": batch_size,
                "Learning_Rate": lr,
                "Total Training Data": f"Size: {1-testset_size}, Total: {len(train_ds)}",
                "Total Validation Data": f"Size: {0.5*(testset_size)}, Total: {len(val_ds)}",
                "Total Evalutaion Data": f"Size: {0.5*(testset_size)}, Total: {len(eval_ds)}",
                "Pytorch GPU": torch.cuda.is_available(),
            })

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    final = trainer.predict(test_dataset=eval_ds)
    trainer.log_metrics("eval", final.metrics)
    trainer.save_metrics("eval", final.metrics)

    final_labels = final.label_ids
    final_predictions = np.argmax(final.predictions, axis=1)
    wandb.log({f"FinalPred_ConfMatrix_{name}" : wandb.plot.confusion_matrix( 
                preds=final_predictions, y_true=final_labels,
                class_names=id2label
        )})
    wandb.log({"Test_Acc": final.metrics['test_accuracy'], 
            "Test_F1": final.metrics['test_f1'],
            "Test_Prec": final.metrics['test_precision'],
            "Test_Recall": final.metrics['test_recall'],
            "Test_Loss": final.metrics['test_loss']})
    wandb.finish(exit_code=0)