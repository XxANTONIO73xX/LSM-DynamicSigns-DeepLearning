# Automatic Recognition of Dynamic Signs of Mexican Sign Language using Deep Learning

![Graphical abstract](/ProjectImages/Graphical%20abstract.png)

This repository provides the implementation of a deep learning approach for the **automatic recognition of dynamic signs in Mexican Sign Language (LSM)**.  
It covers dataset preparation, model training, hyperparameter optimization, and evaluation using state-of-the-art neural network architectures.

---

## 📂 Project Structure

```
├── Data/                     # Dataset and exploratory notebooks
│   ├── ExploreData.ipynb     # here you can visualize the data.
│   └── holistic.mp4
│
├── Helpers/                  # Utility modules for data processing and model definitions
│   ├── data.py
│   ├── models.py
│   └── preprocessing.py      # Code for preprocessing
│
├── Op results/               # Results from optimization experiments
│   ├── resnet_opt_1          # Here you can find the models
│   ├── resnet_opt_2
│   ├── resnet_opt_4          # Best model (F1_score=0.925)
│   └── ...
│
├── ProjectImages/             # Figures for documentation and reports
│   └── Graphical abstract.png
│
├── Experiments.ipynb          # Main notebook for training and evaluation
├── Optimization.ipynb         # Hyperparameter search with Keras Tuner
├── requirements.txt           # Project dependencies
├── LICENSE                    # Project license
└── README.md                  # Project documentation
```

---

## ⚙️ Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Main Libraries
- MediaPipe
- TensorFlow / Keras 
- Keras Tuner  
- TensorFlow Addons  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- tqdm

---

## 📥 Dataset

The project requires a preprocessed dataset to be placed inside the `Data/` directory.  
In [`Experiments.ipynb`](/Experiments.ipynb), the following code is included to download it automatically:

```python
!pip install -q gdown  # Install gdown to handle large Drive files
import gdown

file_id = "1knZzpGblTER4O2KVjXT1ei0uooWGQSTO"
gdown.download(id=file_id, output="./Data/Dataset.csv", quiet=False)
```

Alternatively, you can download the dataset manually from  
[Google Drive](https://drive.google.com/file/d/1knZzpGblTER4O2KVjXT1ei0uooWGQSTO/view?usp=sharing)  
and place it at:

```
/Data/Dataset.csv
```

> ⚠️ **Important:** Do not rename the dataset file, otherwise the notebooks may not run correctly.

---

## 📝 License

This project is licensed under the terms specified in the [LICENSE](/LICENSE) file.  
Recommended for academic/research use: **MIT** or **Apache 2.0**.

---

## ✨ Acknowledgments

This repository is part of the supplementary materials for the manuscript:  

**"Automatic Recognition of Dynamic Signs of Mexican Sign Language using Deep Learning"**  
currently under preparation for submission to *IEEE Latin America Transactions*.
