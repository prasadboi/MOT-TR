# train.py
# Description: Main script to run the DETR fine-tuning process.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
# from tqdm.notebook import tqdm  # Use notebook version if in Colab/Jupyter
from tqdm import tqdm  # Use standard version otherwise

# Import modules from the project
import config
from utils import get_image_paths_from_annotation
from eval import evaluate_detection, visualize_predictions
from dataset import MovedObjectDataset, collate_fn
from model import load_processor, load_model

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou




def main():
    """Main function to execute the training pipeline."""
    print("Starting DETR Fine-tuning Pipeline...")
    print(f"Using device: {config.DEVICE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGGING_DIR, exist_ok=True)

    image_processor = load_processor()

    print("\nPreparing dataset file list...")
    try:
        all_annotation_files = [
            f for f in os.listdir(config.ANNOTATION_DIR) if f.endswith(".txt")
        ]
        # print(all_annotation_files)
        if not all_annotation_files:
            raise ValueError(
                f"No annotation files (.txt) found in {config.ANNOTATION_DIR}"
            )
            
        print(f"Found {len(all_annotation_files)} total annotation files.")
    except FileNotFoundError:
        print(f"Error: Annotation directory not found at {config.ANNOTATION_DIR}")
        return  # Exit if data not found
    except ValueError as e:
        print(f"Error: {e}")
        return
    all_annotation_files = all_annotation_files[:10] # for testing
    # Split files
    train_files, test_files = train_test_split(
        all_annotation_files,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_SEED,
    )
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

    # Create Dataset instances
    train_dataset = MovedObjectDataset(
        annotation_files=train_files,
        annotation_dir=config.ANNOTATION_DIR,
        image_dir=config.IMAGE_DATA_DIR,
        image_processor=image_processor,
        # target_size=config.IMAGE_TARGET_SIZE,
    )
    test_dataset = MovedObjectDataset(
        annotation_files=test_files,
        annotation_dir=config.ANNOTATION_DIR,
        image_dir=config.IMAGE_DATA_DIR,
        image_processor=image_processor,
        # target_size=config.IMAGE_TARGET_SIZE,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # Quick check of one dataset item
    print("\nChecking one dataset item from training set...")
    try:
        # Find first valid item if needed
        sample = None
        first_valid_idx = -1
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            # Check if it's a valid item (pixel_values is not None)
            if sample and sample.get("pixel_values") is not None:
                first_valid_idx = i
                if(first_valid_idx > 2):    
                    break  # Found the first valid one

        if sample and sample.get("pixel_values") is not None:
            print(f"(Sampled item index {first_valid_idx})")
            print("  Sample pixel_values shape:", sample["pixel_values"].shape)
            print("  Sample labels:", sample["labels"]["class_labels"].shape)
            print("  Sample labels shape:", sample["labels"]["class_labels"].shape)
            print("  Sample boxes:", sample["labels"]["boxes"])
            print("  Sample boxes shape:", sample["labels"]["boxes"].shape)
        else:
            print("Could not load a valid sample item after checking initial items.")
            if len(train_dataset) > 0:
                print(
                    "This might indicate issues with file paths or image loading for all checked samples."
                )
            else:
                print("Training dataset appears empty.")

    except Exception as e:
        print(f"Error loading/checking first sample: {e}")
        # import traceback
        # traceback.print_exc()

    model = load_model()
    model.to(config.DEVICE)

    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=20,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 5},
        warmup_ratio=0.1,
        max_grad_norm = 1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Default metric if no compute_metrics provided
        greater_is_better=False,  # Lower eval_loss is better
        remove_unused_columns=False,  # Important for custom collate_fn
        fp16=False, # torch.cuda.is_available(),   # Optional: Enable mixed precision
        report_to="tensorboard",
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        seed=config.RANDOM_SEED,
    )

    train_dataset[0]
    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=image_processor,  # Pass processor for saving purposes (saves config)
        # compute_metrics=compute_eval_metrics, # Define if needed
    )

    # 5. Start Training
    print("\nStarting training...")
    try:
        # for p in model.model.encoder.parameters(): p.requires_grad = False
        # for p in model.model.decoder.parameters(): p.requires_grad = False
        # training_args.num_train_epochs = 20
        # training_args.learning_rate = 1e-4
        # trainer.train()
        for p in model.model.encoder.parameters(): p.requires_grad = True
        for p in model.model.decoder.parameters(): p.requires_grad = True
        train_results = trainer.train()
        print("Training finished.")

        # Save final model, metrics, and state
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        print("Training finished.")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return  # Stop execution

    # 6. Evaluation
    print("\nStarting final evaluation on the test set...")
    try:
        eval_metrics = trainer.evaluate()
        print("Evaluation finished.")
        print("Evaluation Metrics (Loss-based):", eval_metrics)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        # import traceback
        # traceback.print_exc()

    
    metrics = evaluate_detection(
        model=model,
        processor=image_processor,
        dataset=test_dataset,
        collate_fn=collate_fn,
        device=config.DEVICE,
        iou_threshold=0.5,     # you can sweep this
        score_threshold=0.58,   # detector confidence cutoff
    )
    print("Evaluation metrics on test set:")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
    print("\nVisualizing and saving predictions on test samples...")
    for i in range(min(20, len(test_dataset))):
        visualize_predictions(
            model=model,
            processor=image_processor,
            dataset=test_dataset,
            device=config.DEVICE,
            idx=i,
            threshold=0.65,
            save_dir=config.VISUALIZATION_DIRECTORY,
        )


if __name__ == "__main__":
    main()
