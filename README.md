# 🚗 Car License Plate Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

## 📖 Project Overview
This project implements a Deep Learning solution to automatically detect and localize license plates on vehicles. Unlike a standard classification task, this is an **Object Detection** regression problem. The model takes an image of a car as input and predicts the coordinates of the bounding box `(xmin, ymin, xmax, ymax)` surrounding the license plate.



## 📊 Dataset
The project uses the **Car Plate Detection** dataset (likely from Kaggle).
* **Input:** Car images (`.png`) of varying sizes.
* **Annotations:** PASCAL VOC format (`.xml` files) containing the boundary coordinates of the plates.
* **Preprocessing:**
    * Images are resized to a fixed dimension of **224x224**.
    * Bounding box coordinates are scaled proportionally to match the resized images.
    * Pixel values are normalized to the range `[0, 1]`.

## 🛠️ Tech Stack
* **Core:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning:** TensorFlow, Keras
* **Data Handling:** NumPy, Pandas, LXML (for XML parsing)
* **Visualization:** Matplotlib, Seaborn

## 🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) built using the Keras Sequential API. It is designed to extract features from the image and regress the 4 coordinate values.



[Image of convolutional neural network architecture diagram]


| Layer Type | Filters/Units | Kernel/Pool | Activation |
| :--- | :---: | :---: | :---: |
| **Conv2D** | 64 | 3x3 | ReLU |
| **MaxPooling2D** | - | 2x2 | - |
| **Dropout** | - | - | 0.1 |
| **Conv2D** | 32 | 3x3 | ReLU |
| **MaxPooling2D** | - | 2x2 | - |
| **Dropout** | - | - | 0.1 |
| **Flatten** | - | - | - |
| **Dense** | 64 | - | ReLU |
| **Output (Dense)**| 4 | - | **Sigmoid** |

*Note: The output layer uses `Sigmoid` activation because the target coordinates are normalized between 0 and 1.*

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/car-plate-detection.git](https://github.com/your-username/car-plate-detection.git)
    cd car-plate-detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow opencv-python matplotlib lxml pandas numpy scikit-learn
    ```

3.  **Data Setup**
    Ensure your data is structured as follows:
    ```text
    /dataset
        /images/ (contains *.png)
        /annotations/ (contains *.xml)
    ```

4.  **Run the Training Script**
    Run the Jupyter Notebook or convert it to a script to start training.

## 📈 Training Results
The model was trained for **50 Epochs** using the **Adam** optimizer and **Mean Squared Error (MSE)** loss function.

* **Final Training Accuracy:** ~95.2%
* **Final Validation Accuracy:** ~80.0%
* **Final Training Loss:** 0.0004
* **Final Validation Loss:** 0.0085

The model demonstrates strong convergence, though the gap between training and validation suggests mild overfitting, which is addressed via Dropout layers.

## 🖼️ Visualization
The project includes visualization code to verify the ground truth and predictions by drawing bounding boxes over the images using OpenCV.

```python
# Example visualization code snippet
image = cv2.rectangle(X[0], (y[0][0], y[0][1]), (y[0][2], y[0][3]), (255, 0, 255))
plt.imshow(image)
plt.show()
