# Machine Failure Prediction Using Support Vector Machines

Machine Learning **Lab 8**: binary machine-failure classification using the **AI4I 2020 Predictive Maintenance Dataset**. The experiment compares Linear SVM, RBF SVM, and a tuned RBF SVM using Scikit-learn.

## Experiments

- Dataset inspection and exploratory data analysis
- Class distribution, histograms, box plots, and correlations
- `StandardScaler` for numerical features and `OneHotEncoder` for `Type`
- Stratified 80:20 train-test split
- Linear and RBF Support Vector Classifiers
- Confusion matrices, accuracy, precision, recall, and F1-score
- 5-fold `GridSearchCV` with **F1-score** for hyperparameter tuning

## Recorded notebook results

| Model | Accuracy | Failure precision | Failure recall | Failure F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 96.60% | 0.000 | 0.000 | 0.000 |
| RBF SVM | 97.20% | 0.875 | 0.206 | 0.333 |
| Tuned RBF SVM | **98.00%** | 0.759 | **0.603** | **0.672** |

**Note:** Only **339 out of 10,000** samples have `Machine failure = 1`. Accuracy alone is misleading; failure-class recall and F1 are important. Results above are recorded outputs of the supplied notebook, not independently retrained in this package.

## Repository layout

```text
machine-failure-prediction-svm/
├── data/
│   └── ai4i2020.csv       # Add the source CSV here
├── notebook/
│   └── Lab_8_SVM_Classification.ipynb
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Download **`ai4i2020.csv`** from the [UCI AI4I 2020 dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and put it inside `data/`. The dataset was not included in the uploaded notebook. It is a **synthetic** predictive-maintenance dataset released under **CC BY 4.0**; attribution: *AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, DOI: 10.24432/C5HS5C*.

## Run

```bash
pip install -r requirements.txt
python main.py
```

For the notebook, open `notebook/Lab_8_SVM_Classification.ipynb` in Jupyter or VS Code, then **Restart & Run All**.

## Author

**Tanushri Sapate**  
B.Tech CSE (Artificial Intelligence & Machine Learning)  
Section B-B4 · Roll 55
