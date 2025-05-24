# 🐱🐶 Cats vs Dogs Image Classification using ResNet-50

This project applies deep learning and transfer learning techniques to classify images of cats and dogs. Leveraging the power of a pre-trained **ResNet-50** model, the solution achieves high classification accuracy on the Kaggle Dogs vs. Cats dataset.

> 📎 **View Full Notebook on Kaggle:**  
> [Cats & Dogs Image Classification | ResNet-50](https://www.kaggle.com/code/lykin22/cats-dogs-image-classification-resnet-50)


---

## 📂 Project Structure

```
cats-dogs-resnet50/
├── data/                 # Directory for training and validation images
├── notebooks/
│   └── cats_dogs_resnet50.ipynb   # Main training and evaluation notebook
├── models/               # Saved model weights (optional)
├── outputs/              # Logs, plots, and predictions
├── requirements.txt      # Dependencies
└── README.md             # Project overview
```

---

## 🧠 Model Overview

* **Architecture:** ResNet-50 (pre-trained on ImageNet)
* **Layers Modified:** Top classification layer replaced with custom dense layer for binary classification
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Metrics:** Accuracy

---

## 🧪 Key Steps

1. **Data Preparation**

   * Downloaded the [Dogs vs. Cats dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data)
   * Resized images to 224x224 to match ResNet-50 input format
   * Applied train-validation split and data augmentation

2. **Model Building**

   * Used Keras `ResNet50` with `include_top=False`
   * Added custom fully-connected layers for binary classification
   * Compiled with `Adam` optimizer and `binary_crossentropy` loss

3. **Training**

   * Used early stopping and model checkpoint callbacks
   * Trained for multiple epochs with real-time data augmentation

4. **Evaluation & Prediction**

   * Evaluated on validation data
   * Visualized confusion matrix, classification report, and sample predictions

---

## 📊 Results

* Achieved **\~98% validation accuracy**
* Model generalizes well across different image types
* Visualizations confirm robust prediction across classes

---

## 🚀 How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/cats-dogs-resnet50.git
cd cats-dogs-resnet50

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the notebook
jupyter notebook notebooks/cats_dogs_resnet50.ipynb
```

---

## 📌 Requirements

* TensorFlow / Keras
* NumPy, Matplotlib, scikit-learn
* Jupyter Notebook
* PIL, tqdm

---

## 📈 Future Improvements

* Hyperparameter tuning
* Experiment with deeper architectures (e.g., EfficientNet, DenseNet)
* Deploy model via Flask or Streamlit

---

## 📎 References

* [Kaggle Dataset: Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats)
* [ResNet-50 Paper](https://arxiv.org/abs/1512.03385)

