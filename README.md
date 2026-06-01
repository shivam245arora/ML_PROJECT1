# Diabetes Prediction Using Machine Learning

## Overview

This project aims to develop a predictive model for diabetes using various machine learning techniques. By leveraging patient data, the model can help identify individuals at risk of developing diabetes, allowing for early intervention and improved health outcomes.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Introduction

Diabetes is a chronic disease that affects millions of people worldwide. Early prediction and diagnosis are crucial for effective management. This project utilizes machine learning algorithms to analyze health data and predict the likelihood of diabetes in individuals.

## Dataset

The dataset used for this project is sourced from the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database). It includes various health metrics such as:

- Pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin
- Body mass index (BMI)
- Diabetes pedigree function
- Age
- Outcome (0 or 1, indicating the presence of diabetes)

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clonehttps://github.com/shivam245arora/ML_PROJECT1/tree/main
   cd diabetes-prediction
   ```

2. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the diabetes prediction model, execute the following command:

```bash
python predict.py
```

You can also modify the input data in `input_data.csv` to test different scenarios.

## Modeling Techniques

This project explores several machine learning algorithms, including:

- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- Neural Networks

The model performance is evaluated using metrics such as accuracy, precision, recall, and F1 score.

## Results

After training and evaluating the models, the Random Forest model achieved the highest accuracy of **85%** on the test dataset. Detailed results can be found in the `results` folder, including confusion matrices and classification reports.

## Conclusion

The project successfully demonstrates the potential of machine learning for predicting diabetes. Future work may involve refining the models, incorporating additional features, or utilizing deep learning techniques for improved accuracy.



## Acknowledgments

- [Kaggle](https://www.kaggle.com) for the dataset.
- [Scikit-learn](https://scikit-learn.org/) for the machine learning library.
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization.

---

Feel free to contribute or reach out with any questions or suggestions!
