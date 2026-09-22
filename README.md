# CSPDarkNet Classification Example

CSPDarkNet image-classification training and ONNX export example. The code covers directory-based datasets, augmentation, training, validation, EMA, early stopping, checkpointing, and optional ONNX simplification.

This is a reusable training example, not a released model or board application. Dataset classes and deployment requirements must be configured for the target project.

## Requirements

The scripts use Python 3 with PyTorch, torchvision, NumPy, OpenCV, Pillow, tqdm, Loguru, ONNX, ONNX Simplifier, THOP, and optionally Weights & Biases.

## Dataset

Training and validation directories use one subdirectory per class:

```text
dataset/
├── train/
│   ├── bicycle/
│   ├── electric_bicycle/
│   └── gastank/
└── val/
    ├── bicycle/
    ├── electric_bicycle/
    └── gastank/
```

The directory names and order must match `ClassificationDataset.class_names` in `dataset.py`. Pass the same ordered values through `trainer.py --class_names`; that argument sets the model/output labels but does not replace the dataset class list.

Images are resized with aspect-ratio-preserving padding and then transformed to the configured square `image_size`.

## Train

```bash
python3 trainer.py \
  --device_id cuda \
  --trainset_path <dataset>/train \
  --valset_path <dataset>/val \
  --class_names bicycle electric_bicycle gastank \
  --batch_size 16 \
  --image_size 640 \
  --max_epochs 300 \
  --ema_enabled \
  --early_stop
```

Add `--wandb_enabled` to write an offline Weights & Biases run. Checkpoints are written under `--save_dir` (default `runs/`); the best checkpoint is `<model_name>.pt` and the latest checkpoint is `<model_name>_latest.pt`.

Use `--debug_mode` for a one-iteration-per-epoch pipeline check. It is not a training-quality validation.

## Export ONNX

```bash
python3 export.py \
  --weight runs/<model>.pt \
  --device cpu \
  --input_shape 1 3 640 640 \
  --input_names image \
  --output_names output \
  --opset_version 13 \
  --enable_onnxsim
```

The exporter prints parameter and FLOP estimates and waits for confirmation before writing `<model>.onnx` beside the checkpoint. Ensure that the class count and ordering used to construct the model match the checkpoint before export.

## Verification

- Check that every configured class has images in both dataset splits.
- Record overall and per-class validation accuracy from `trainer.py`.
- Load the exported ONNX with an ONNX runtime or checker used by the target deployment flow.
- Compare the exported model output with the PyTorch checkpoint on fixed samples before deployment conversion.

No accuracy target or deployment-platform compatibility is asserted by this repository.

## Files

```text
model.py              CSPDarkNet model definition
dataset.py            classification dataset and class ordering
trainer.py            training and validation entry point
export.py             ONNX export entry point
demo.py               single-image PyTorch inference example
dataset_visualizer.py dataset inspection helper
```
