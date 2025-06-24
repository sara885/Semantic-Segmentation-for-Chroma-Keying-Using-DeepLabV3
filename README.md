# Chroma Key Segmentation with DeepLabV3

## Project Description  
This project implements semantic segmentation for chroma keying using DeepLabV3 with a ResNet-50 backbone. The model is trained on the AIM-500 dataset to separate the foreground from the background using trimap masks. Transfer learning is leveraged to optimize training efficiency and achieve high accuracy. This approach is suitable for applications in film production and virtual environments.

---

## Dataset  
- **AIM-500 Dataset**: Includes RGB images paired with trimap masks that segment the foreground and background areas.  
- **Input Images Location**: `/original` folder  
- **Masks Location**: `/trimap` folder

---

## Model Details  
- Architecture: DeepLabV3 with ResNet-50 backbone  
- Final classifier adapted for 2 classes: foreground and background  
- Backbone layers are frozen to speed up training and reduce overfitting

---

## Training  
- Loss Function: CrossEntropyLoss  
- Optimizer: AdamW with learning rate 0.0001  
- Number of Epochs: 15  
- Evaluation Metric: Intersection over Union (IoU)  
- Final Training Results:  
  - Loss = 0.0952  
  - IoU = 0.9092

---

## Inference and Visualization  
- Predict segmentation masks on new images  
- Visualize original images alongside predicted masks using matplotlib for easy comparison

---

## How to Run

1. Clone the repository:  
```bash
git clone https://github.com/your-username/chroma-key-segmentation.git
cd chroma-key-segmentation
```

---
2. Install the required libraries:
```bash
pip install torch torchvision matplotlib numpy pillow
```
---
3.Update the image and mask paths in the script if necessary.

---
4.Run the training and inference script:
```bash
python train_and_predict.py
```
---

## Future Work:

Extend support for 3-class trimaps (foreground, background, unknown region)

Integrate real-time video segmentation capabilities

Export the trained model for deployment formats such as ONNX and TFLite

## License:
This project is licensed under the MIT License.

## Author:
Developed by Sara Issawi

