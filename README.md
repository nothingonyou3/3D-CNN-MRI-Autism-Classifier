# MRI Autism Classification — 3D CNN

University project developed for the **Artificial Intelligence course (2022/2023)**.

**Authors:**  Giulia Avanzato (70/90/00356), Samuele Capacci (70/90/00341)

## Description

The goal of this project is to classify **3D Magnetic Resonance Imaging (MRI)** scans, distinguishing between subjects diagnosed with **Autism Spectrum Disorder (ASD)** and control subjects using a **3D Convolutional Neural Network (3D CNN)** implemented in **PyTorch**.

The dataset consists of approximately **900 three-dimensional MRI scans** collected from different research centers. All images have dimensions of `61 × 73 × 61` and contain a single grayscale channel.

## Project Structure

```text
.
├── mri_datautils.py   # Custom dataset, NIfTI loading/cropping, label generation
├── dataset_utils.py   # Train/test split and k-fold cross-validation
├── models.py          # TriConvNet and TriConvNet2 architectures
├── trainings.py       # Training and accuracy evaluation functions
├── main.py            # Main script: dataset preparation, DataLoaders, training
└── README.md
```

### `mri_datautils.py`

This module provides the utilities required to load and preprocess the MRI data.

* **`NiiDataset`**: custom dataset extending `torch.utils.data.Dataset`, used to load `.nii` images and their corresponding labels from a CSV file.
* **`read_nii`**: loads a `.nii` file using [NiBabel](https://nipy.org/nibabel/), converts it into a PyTorch tensor, adds the channel dimension (expected format: `(C, D, H, W)`), and crops empty regions of the image.
* **`plot_nii_images`**: plots the three central slices of an MRI scan, one along each axis, to visually verify that the cropping operation does not remove relevant information.
* **`generate_label`**: generates the CSV file containing the labels (`filename, label`) required by the dataset.

### `dataset_utils.py`

Provides utilities for creating training and evaluation splits.

* **`train_test_split_dataset`**: creates separate `NiiDataset` instances for training and testing according to a specified split ratio.
* **`kfold_split_dataset`**: generates `k` train/test dataset pairs for k-fold cross-validation. This functionality was implemented but not used in the final experiments because of the computational cost.

### `models.py`

Two 3D convolutional neural network architectures are implemented using combinations of `nn.Sequential` modules.

#### TriConvNet

The complete architecture is inspired by [3D-CNN-PyTorch (C3DNet)](https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py).

It consists of:

* 6 convolutional layers (`Conv3d` + `BatchNorm3d` + `ReLU`)
* `MaxPool3d` layers in selected blocks
* 2 upsampling layers (`Upsample`, scale factor 2)
* 3 fully connected layers (`Linear` + `ReLU` + `Dropout`)

#### TriConvNet2

A simplified version of `TriConvNet`, consisting of:

* 4 convolutional layers
* 2 fully connected layers

Both architectures expose a `scale()` method used during development to inspect the model size and number of parameters.

### `trainings.py`

Contains the main training and evaluation functions.

* **`train_and_save`**: trains a model for a specified number of epochs using the selected loss function and optimizer, evaluates accuracy after each epoch, and optionally saves the best-performing model weights.
* **`check_accuracy`**: computes classification accuracy by comparing predicted classes with the ground-truth labels.

The training implementation is based on the official PyTorch [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial.

### `main.py`

The main entry point of the project.

The script:

1. Generates the label CSV file.
2. Creates the training and test splits.
3. Initializes the `DataLoader` objects.
4. Trains both CNN architectures.

A batch size of **32** is used to avoid loading the entire 3D dataset into memory at once.

## Requirements

The project requires:

```text
torch
nibabel
numpy
matplotlib
```

Install the dependencies with:

```bash
pip install torch nibabel numpy matplotlib
```

## Usage

1. Place the `.nii` MRI scans in a directory, for example:

```text
data/mri_images/
```

2. If necessary, modify the label-generation logic in `generate_label` (`mri_datautils.py`) according to the naming convention used by the dataset.

3. Configure the dataset paths in `main.py`:

```python
DATA_DIR = ...
LABELS_CSV = ...
```

4. Run the training script:

```bash
python main.py
```

The model weights corresponding to the best observed accuracy are saved in the `weights/` directory.

## Results

Several experiments were conducted by varying the **learning rate** (`0.001` and `0.01`) and the **number of training epochs** (up to 200) for both architectures.

| Model       | Learning Rate | Epochs | Observed Accuracy |
| ----------- | :-----------: | :----: | :---------------: |
| TriConvNet  |     0.001     |  2–200 |     ~39% – 66%    |
| TriConvNet  |      0.01     |  5–10  |     ~49% – 60%    |
| TriConvNet2 |     0.001     |  2–200 |     ~38% – 57%    |
| TriConvNet2 |      0.01     |  5–10  |     ~52% – 55%    |

Although the training loss progressively decreased, classification accuracy did not consistently exceed **55%**.

Since the task is a balanced binary classification problem, random classification provides a baseline of approximately **50% accuracy**. The observed results therefore indicate that the implemented architectures were not able to learn a sufficiently discriminative representation from the available MRI data.

## Future Improvements

Possible directions for improving the project include:

* Redesigning the CNN architectures, including network depth, number of filters, and pooling strategy.
* Experimenting with different optimizers and loss functions.
* Applying data augmentation techniques specifically designed for 3D MRI data.
* Using the implemented k-fold cross-validation to obtain a more robust estimate of model performance.
* Improving voxel-intensity normalization and standardization.
* Exploring transfer learning from pretrained 3D CNN architectures.
* Investigating alternative approaches for handling the multi-center nature of the dataset.

## References

* [PyTorch — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [PyTorch — Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* [PyTorch — BatchNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html)
* [PyTorch — Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
* [PyTorch — Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
* [3D-CNN-PyTorch (C3DNet)](https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py)
* [NiBabel](https://nipy.org/nibabel/)
