# Support Vector Machine on Breast Cancer Dataset

This project demonstrates how to use a Support Vector Machine (SVM) classifier from scikit-learn to classify tumors as malignant or benign using the classic Breast Cancer dataset.

## Features

- Loads the Breast Cancer dataset from scikit-learn
- Preprocesses the data (feature scaling)
- Splits data into training and test sets
- Trains an SVM classifier
- Evaluates the model with a confusion matrix and classification report
- Visualizes a 2D projection of predictions

## Requirements

- Python 3.x
- scikit-learn
- numpy
- matplotlib

Install the requirements with:

```bash
pip install scikit-learn numpy matplotlib
```

## Usage

Run the script:

```bash
python svm_breast_cancer.py
```

You will see a classification report, confusion matrix, and a 2D scatter plot of predictions.

## File Structure

- `svm_breast_cancer.py` — Main script for SVM classification
- `README.md` — Project documentation

## License

This project is licensed under the MIT License.