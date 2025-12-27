# Bird Species Classification with PyTorch

This project focuses on building and training deep learning models to classify bird species from images using **PyTorch**. It includes two different training approaches:

1. A **custom convolutional neural network (CNN)** built from scratch.
2. A **pretrained transfer learning model** based on **ResNet-50**.

The goal is to compare a fully custom model with a modern pretrained architecture and understand the tradeoffs between them in terms of accuracy, training time, and generalization.

---

## Project Structure

```
├── model.py # Custom CNN architecture
├── pretrained_model.py # ResNet50-based transfer learning model
├── train.py # Training loop and dataset loading
├── predict.py # Run inference on a single image
├── bird_cnn_[retrained].pth # Saved model checkpoint
└── README.md
```

---

## Models

### 1. Custom CNN (`model.py`)

This model is implemented entirely from scratch using standard convolutional building blocks:

- Convolution layers
- Batch normalization
- ReLU activations
- Max pooling
- Fully connected classifier

The architecture progressively increases feature depth (32 → 256 channels) while reducing spatial resolution. Images are expected to be resized to **128×128**, which results in a final feature map of **8×8** before classification.

This model is useful for:
- Understanding how CNNs work internally
- Learning how architectural choices affect performance
- Training when pretrained weights are not desired

---

### 2. Pretrained Model (`pretrained_model.py`)

This model uses **ResNet-50**, pretrained on **ImageNet**, as a feature extractor. The original classification head is replaced with a custom classifier tailored to the bird species dataset.

Key ideas here:
- The backbone has already learned general visual features (edges, textures, shapes).
- Only the final layers need to adapt to bird species classification.
- Training is faster and typically more accurate with smaller datasets.

This approach demonstrates **transfer learning**, which is widely used in real-world machine learning systems.

---

## Machine Learning Concepts Used

### Convolutional Neural Networks (CNNs)
CNNs are designed to process images by learning spatial hierarchies of features. Early layers learn simple patterns like edges, while deeper layers capture more complex structures.

### Transfer Learning
Instead of training a large network from scratch, we reuse a pretrained model and fine-tune it. This:
- Reduces training time
- Improves performance on limited data
- Leverages knowledge from large-scale datasets

### Freezing and Unfreezing Layers
During training with a pretrained model:
- **Frozen layers** do not update their weights and act as fixed feature extractors.
- **Unfrozen layers** are trainable and adapt to the new task.

This helps prevent overfitting early in training and allows controlled fine-tuning later.

### Overfitting Prevention
Several techniques are used to improve generalization:
- Dropout in the classifier
- Data normalization and resizing
- Validation set monitoring

---

## Dependencies
This project requires the following Python packages:
- Python 3.8+
- PyTorch
- torchvision
- Pillow
- NumPy

### Install dependencies

Using `pip`:
```
pip install torch torchvision pillow numpy
```

If you have an NVIDIA GPU, make sure your PyTorch installation matches your CUDA version. You can verify GPU support with:
```
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Training

### Train from scratch (custom CNN)
```
python train.py
```

Training scripts load image datasets using `more_data`, apply transformations, and optimize the model using cross-entropy loss.

Models are trained on GPU when available:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Checkpoints store:
- Model weights
- Class-to-index mapping
- Class names for inference

---

## Inference

The predict.py script loads a saved checkpoint and runs inference on a single image. The output is the predicted bird species based on the trained classifier.

Example:
```
python predict.py path/to/image.jpg
```

The output will include a list of probable birds ranked by the softmax evaluator.

---

## Summary

This project demonstrates both foundational and modern approaches to image classification:
- Building a CNN from scratch to understand the fundamentals
- Applying transfer learning for better real-world performance
- Using PyTorch best practices for training and inference
- Together, these models provide a solid introduction to practical computer vision workflows.

---

## Dataset Acknowledgement

This project uses bird species image data for training and evaluation.  
Huge thanks to the authors and maintainers of the following public dataset repository:
**[UmairPirzada/BIRDS_SPECIES--IMAGE-CLASSIFICATION](https://github.com/UmairPirzada/BIRDS_SPECIES--IMAGE-CLASSIFICATION)**
