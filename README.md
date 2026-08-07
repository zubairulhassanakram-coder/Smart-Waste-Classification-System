# Smart Waste Classification System using CNN

A Convolutional Neural Network (CNN) based image classifier that identifies waste
material into three categories: **Cardboard**, **Glass**, and **Metal**.

This project was built as a university assignment to demonstrate a complete,
beginner-friendly deep learning pipeline — from data preprocessing to model
training and evaluation.

---

## Dataset

**TrashNet Dataset** (subset — 3 classes only)
Source: https://www.kaggle.com/datasets/feyzazkefe/trashnet

| Class      | Description                  |
|------------|-------------------------------|
| Cardboard  | Images of cardboard waste     |
| Glass      | Images of glass waste         |
| Metal      | Images of metal waste         |

**Setup instructions:**
1. Download the dataset from the Kaggle link above.
2. Extract only the `cardboard`, `glass`, and `metal` folders.
3. Place them inside the `dataset/` folder, matching this structure:

```
dataset/
├── cardboard/
├── glass/
└── metal/
```

> Note: The `dataset/` folder is excluded from Git via `.gitignore` due to file size.

---

## Project Structure

```
Smart-Waste-Classification-System/
│
├── dataset/                 # TrashNet images (cardboard, glass, metal)
├── notebooks/                # Jupyter notebooks for each pipeline stage
├── models/                   # Saved trained model files
├── images/                   # Output plots and sample predictions
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_preprocessing.ipynb` | Load images, resize, normalize, split into train/val/test |
| `02_model_training.ipynb` | Build and train the CNN model |
| `03_model_evaluation.ipynb` | Evaluate model, plot accuracy/loss, show sample predictions |

---

## How to Run

1. Clone/download this project.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the project in VS Code.
5. Run the notebooks in order:
   - `01_data_preprocessing.ipynb`
   - `02_model_training.ipynb`
   - `03_model_evaluation.ipynb`

---

## Model

A simple CNN architecture (Conv2D → MaxPooling → Dense layers) built with
TensorFlow/Keras, trained to classify images into 3 waste categories.

---

## Author

University CNN Assignment — Smart Waste Classification System
