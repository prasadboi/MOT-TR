# train.py
# Description: Main script to run the DETR fine-tuning process.

import os
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader  # For optional detailed evaluation

# from tqdm.notebook import tqdm  # Use notebook version if in Colab/Jupyter
from tqdm import tqdm  # Use standard version otherwise

# Import modules from the project
import config
import utils
from dataset import MovedObjectDataset, collate_fn
from model import load_processor, load_model

# Optional: For detailed evaluation metrics
# from torchmetrics.detection import MeanAveragePrecision
# from utils import format_predictions_for_metrics, format_gts_for_metrics # Need to implement these


def main():
    """Main function to execute the training pipeline."""
    print("Starting DETR Fine-tuning Pipeline...")
    print(f"Using device: {config.DEVICE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGGING_DIR, exist_ok=True)

    # 1. Load Image Processor
    image_processor = load_processor()

    # 2. Prepare Dataset List
    print("\nPreparing dataset file list...")
    try:
        all_annotation_files = [
            f for f in os.listdir(config.ANNOTATION_DIR) if f.endswith(".txt")
        ]
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
                break  # Found the first valid one

        if sample and sample.get("pixel_values") is not None:
            print(f"(Sampled item index {first_valid_idx})")
            print("  Sample pixel_values shape:", sample["pixel_values"].shape)
            print("  Sample labels:", sample["labels"])
            print("  Sample labels shape:", sample["labels"].shape)
            print("  Sample boxes:", sample["boxes"])
            print("  Sample boxes shape:", sample["boxes"].shape)
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

    # 3. Load Model
    model = load_model()
    model.to(config.DEVICE)

    # 4. Training Setup
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
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Default metric if no compute_metrics provided
        greater_is_better=False,  # Lower eval_loss is better
        remove_unused_columns=False,  # Important for custom collate_fn
        fp16=torch.cuda.is_available(),   # Optional: Enable mixed precision
        report_to="tensorboard",  # Logging backend
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        seed=config.RANDOM_SEED,
    )

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
        train_results = trainer.train()
        print("Training finished.")

        # Save final model, metrics, and state
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        # import traceback
        # traceback.print_exc()
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

    # --- Optional: Detailed Evaluation (Example Sketch) ---
    # print("\nStarting detailed evaluation (mAP)...")
    # try:
    #     # Ensure model is in eval mode and on correct device
    #     model.eval()
    #     model.to(config.DEVICE)
    #
    #     all_preds_formatted = []
    #     all_gts_formatted = []
    #     # Create a DataLoader for the test set
    #     test_dataloader = DataLoader(
    #         test_dataset,
    #         batch_size=config.BATCH_SIZE, # Use eval batch size
    #         collate_fn=collate_fn,
    #         num_workers=config.DATALOADER_NUM_WORKERS
    #     )
    #
    #     # Instantiate metric calculator (e.g., torchmetrics)
    #     # metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox') # Adjust format if needed
    #
    #     with torch.no_grad():
    #         for batch in tqdm(test_dataloader, desc="Detailed Eval"):
    #             if not batch or not batch["pixel_values"].numel(): continue # Skip empty/invalid batches
    #
    #             pixel_values = batch["pixel_values"].to(config.DEVICE)
    #
    #             # Perform inference
    #             outputs = model(pixel_values=pixel_values)
    #
    #             # Post-process to get bounding boxes, scores, labels
    #             # The target_sizes should correspond to the size *before* processor's final resize,
    #             # often the size input to the processor. Here, it's based on diff_pil.
    #             # However, post_process_object_detection usually expects original image size before *any* resizing.
    #             # This needs careful handling. Let's assume processor handles it based on its internal logic
    #             # or we might need to pass the original image size if known and consistent.
    #             # Using IMAGE_TARGET_SIZE as a proxy, but might need adjustment.
    #             orig_sizes = torch.tensor([config.IMAGE_TARGET_SIZE] * pixel_values.shape[0]).to(config.DEVICE)
    #             results = image_processor.post_process_object_detection(
    #                 outputs, threshold=config.EVAL_THRESHOLD, target_sizes=orig_sizes
    #             )
    #
    #             # Format predictions and ground truths for the metric library
    #             # preds_formatted = format_predictions_for_metrics(results) # Implement this
    #             # gts_formatted = format_gts_for_metrics(batch['labels'], batch['boxes']) # Implement this
    #
    #             # metric.update(preds_formatted, gts_formatted)
    #             # Or store them:
    #             # all_preds_formatted.extend(preds_formatted)
    #             # all_gts_formatted.extend(gts_formatted)
    #
    #     # Compute final metrics
    #     # final_metrics = metric.compute()
    #     # print("Detailed Evaluation Metrics (e.g., mAP):", final_metrics)
    #
    # except Exception as e:
    #     print(f"Error during detailed evaluation: {e}")
    #     # import traceback
    #     # traceback.print_exc()

    print("\nPipeline finished successfully.")
    print(f"Fine-tuned model and results saved to: {config.MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
