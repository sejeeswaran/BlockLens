import argparse
import os
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

def train_model(dataset_path, model_name="google/vit-base-patch16-224", output_dir="./blocklens_model", epochs=3):
    """
    Fine-tunes a Vision Transformer (ViT) model on a custom dataset.
    
    Args:
        dataset_path (str): Path to the dataset folder (should have 'train' and 'test' subfolders, 
                            or class folders like 'real' and 'fake').
        model_name (str): The pre-trained model to start from.
        output_dir (str): Where to save the trained model.
        epochs (int): Number of training epochs.
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load dataset using ImageFolder structure
    # Expected structure:
    # dataset_path/
    #   train/
    #     real/
    #     fake/
    #   test/
    #     real/
    #     fake/
    try:
        dataset = load_dataset("imagefolder", data_dir=dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure your dataset has 'train' and 'test' folders, or just class folders.")
        return

    # Prepare label mappings
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    print(f"Labels found: {labels}")

    # Load Image Processor
    processor = ViTImageProcessor.from_pretrained(model_name)

    # Define transforms
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    
    _train_transforms = Compose(
        [
            RandomResizedCrop(processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(processor.size["height"]),
            CenterCrop(processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    # Apply transforms
    if "train" in dataset:
        dataset["train"].set_transform(train_transforms)
    if "test" in dataset:
        dataset["test"].set_transform(val_transforms)
    elif "validation" in dataset:
        dataset["validation"].set_transform(val_transforms)

    # Load Model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='none',
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test") or dataset.get("validation"),
        tokenizer=processor,
        data_collator=collate_fn,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Training complete!")

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Deepfake Detection Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224", help="Base model to fine-tune (e.g., 'google/vit-base-patch16-224' or 'Organika/sdxl-detector')")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="./blocklens_model", help="Output directory")

    args = parser.parse_args()
    
    train_model(args.dataset, args.model, args.output, args.epochs)
