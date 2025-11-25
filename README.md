# Razorback Image Classifier (CNN)

This project trains a Convolutional Neural Network (CNN) to classify images as **OfficialRazorback** vs **Other** using PyTorch.  
All images are resized to **500×500** pixels and fed into a custom CNN defined and trained in the Jupyter notebook **`Hogs_CNN.ipynb`**.

---

## Project Structure


```bash
CNN_CLASSIFICATION/
├── Hogs_CNN.ipynb
├── README.md
└── CNN_Data/
    └── CNN/
        ├── official_razorback/
        │   ├── img1
        │   ├── img2
        │   └── ... img30
        └── other/
            ├── img1
            ├── img2
            └── ... img30
```
---

## Requirements

This project requires:

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pandas

---

## Model Details

The model is a custom CNN built with PyTorch, roughly:

        Input: 3-channel RGB images, 500×500.

        Convolutional layers:

        Conv → ReLU → MaxPool

        Conv → ReLU → MaxPool

        Conv → ReLU → MaxPool

        After 3 pooling operations:

        Spatial size goes: 500 → 250 → 125 → 62

        Feature map: 128 × 62 × 62 (for the final conv layer)

        Fully connected (dense) layers:

        Flatten → Linear → Dropout → Linear → Output logits for 2 classes.
