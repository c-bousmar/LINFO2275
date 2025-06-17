# Three-Dimensional Hand Gestures Classification

## A Comparative Study of Machine Learning Techniques

This project implements and compares various machine learning techniques for 3D hand gesture classification. Methods range from baseline approaches like KNN with different distance metrics to advanced techniques such as LSTM, HMM, and DenseNet.

## Project Structure

```
Project2/
├── Data/                 # Raw and preprocessed data
│   └── Dataset_csv/      # Base data
├── Resources/            # Instructions and reference papers
├── Results/              # Experimental results in CSV format
├── src/                  # Source code
│   ├── advanced/         # Advanced algorithms
│   │   ├── DenseNetGestureRecognizer.py
│   │   ├── DollarOne3DGestureRecog...
│   │   ├── FastMDSWithClassifier.py
│   │   ├── HMMClassifier.py
│   │   ├── LRGestureRecognizer.py
│   │   └── LSTMGestureRecognizer.py
│   ├── baseline/         # Baseline algorithms
│   │   ├── distance_metrics.py
│   │   ├── KNN_Classifier.py
│   │   └── KNNWithCustomDist.py
│   ├── datasets_utils.py # Utilities for dataset manipulation
│   ├── GestureRecognizerEstimator.py # Generic evaluation class
│   └── label_conversion.py # Clustering and sequence conversion
├── .gitignore
├── LINFO2275-Report-P2-Group10.pdf # Report
└── README.md
```

## Usage

### Data Preprocessing

Preprocessing is performed using the `datasets_utils.py` and `label_conversion.py` modules.

### Using Classifiers

All classifiers follow the same API with `fit()` and `predict()` methods:

```python
# Example with FFNN
from advanced.DenseNetGestureRecognizer import DenseNetGestureRecognizer

# Initialize the classifier
ffnn = DenseNetGestureRecognizer(epochs=100, batch_size=32, verbose=True)

# Train the model
ffnn.fit(X_train, y_train)

# Predict classes
predictions = ffnn.predict(X_test)
```

### Running Evaluations

Each classifier file contains a main function that runs both user-dependent and user-independent validations and saves the results as CSV files:

```bash
# Make sure to run from the src/ directory for relative paths to work
cd src/

# Run a baseline classifier
python3 baseline/KNN_Classifier.py

# Run an advanced classifier
python3 advanced/LSTMGestureRecognizer.py
```

### Model Evaluation

The `GestureRecognizerEstimator` class allows for evaluating the performance of different algorithms:

```python
from GestureRecognizerEstimator import GestureRecognizerEstimator
from advanced.DenseNetGestureRecognizer import DenseNetGestureRecognizer

# Initialize the estimator
evaluator = GestureRecognitionEvaluator(verbose=True)
    
# User-independent evaluation
results_indep = evaluator.evaluate(
    model=DenseNetGestureRecognizer(epochs=100, batch_size=32, verbose=True),
    df=df,
    evaluation_type="user-independent",
    normalize=True
)

# User-dependent evaluation
results_dep = evaluator.evaluate(
    model=DenseNetGestureRecognizer(epochs=100, batch_size=32, verbose=True),
    df=df,
    evaluation_type="user-dependent",
    normalize=True
)

# Save results
estimator.evaluator.save_results_to_csv(results_indep, f"../Results/DenseNet/_user_independent_all_domain.csv")
evaluator.save_results_to_csv(results_dep, f"../Results/DenseNet/_user_dependent_all_domain.csv")
```

## Implemented Algorithms

### Baseline
- **KNN** with different distance metrics:
  - Euclidean
  - Dynamic Time Warping (DTW)
  - Levenshtein (Edit Distance)
  - Longest Common Subsequence (LCS)

### Advanced
- **LSTM** (Long Short-Term Memory)
- **HMM** (Hidden Markov Model)
- **DenseNet** (Convolutional Neural Network)
- **$1 Recognizer** adapted for 3D
- **Fast MDS with Classifier** (Multidimensional Scaling)
- **LR** (Logistic Regression)

## Validation Methods

Two types of validation are implemented:
1. **User-dependent**: The model is trained and tested on data from the same user
2. **User-independent**: The model is trained on data from certain users and tested on others

## Results

Results from all methods are saved as CSV files in the `Results/` folder. To analyze performances:

## Running the Code

Important note: All scripts should be run from the `src/` directory to ensure relative paths work correctly.

Each classifier file includes a main function that:
1. Loads and preprocesses the dataset
2. Runs both user-dependent and user-independent validations
3. Saves the results as CSV files in the `Results/` folder

## Authors

- Cyril Bousmar – SINF – cyril.bousmar@uclouvain.be  
- Mathis Delsart – INFO – mathis.delsart@student.uclouvain.be  
- Sienou Lamien – SINF – sienou.lamien@student.uclouvain.be