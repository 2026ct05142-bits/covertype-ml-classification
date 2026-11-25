# Forest Cover Type Classification: ML Assignment

## a. Problem Statement
Predict the type of forest cover (7 types) from map data like elevation, slope, distance to water and roads, and soil type. I trained 5 classification models on the same data and compared them using 6 evaluation metrics to find the best one.

## b. Dataset Description
- Source: UCI Machine Learning Repository, Covertype dataset
- Rows: 581,012 in total. I used a random sample of 50,000 rows with the same class proportions, because the full data is too slow for kNN and Random Forest.
- Features: 54 (10 numeric, 4 wilderness-area, 40 soil-type)
- Target variable: Cover_Type (7 classes)
- Feature types: 10 numeric columns, and 44 binary (0 or 1) columns
- Why I chose it: it has many rows and features, it has 7 classes, and the classes are not balanced. It meets the assignment rules of at least 12 features and 1,000 rows.
- Missing values: none found, so no handling was needed
- Duplicates: none found
- Encoding: not needed, because the wilderness and soil columns are already 0 or 1
- Scaling: StandardScaler on the 10 numeric columns, fitted on the training data only and then applied to the test data
- Train-test split: 80:20, same class proportions in both parts, random seed 42 (40,000 train rows, 10,000 test rows)
- Class distribution: the classes are imbalanced. Class 2 is the biggest (about 49%) and class 4 is the smallest (about 0.5%).

| Class | Rows in full data |
|---|---|
| 1 | 211,840 |
| 2 | 283,301 |
| 3 | 35,754 |
| 4 | 2,747 |
| 5 | 9,493 |
| 6 | 17,367 |
| 7 | 20,510 |

## c. Github Repository Link
https://github.com/2026ct05142-bits/covertype-ml-classification

Files in the repository:
- README.md
- requirements.txt
- test_data.csv (the 10,000 test rows)
- results.csv (the metrics table)
- metrics_comparison.png (the graph)
- model/ml_classification.ipynb (all the code)
- model/*.pkl (the 5 saved models and the scaler)

How to run:
1. Run: pip install -r requirements.txt
2. Download covtype.data from UCI and keep it in the same folder as the notebook.
3. Open model/ml_classification.ipynb and run all cells from top to bottom.

## d. Models Used

Models: Logistic Regression, Decision Tree, kNN, Naive Bayes (Gaussian) and Random Forest (Ensemble).

Model settings:
- Logistic Regression: max_iter = 1000
- Decision Tree: default settings, random seed 42
- kNN: 5 neighbours
- Naive Bayes: Gaussian, default settings
- Random Forest: 30 trees, random seed 42

Precision, Recall and F1 use weighted averaging because there are 7 classes. AUC is one-vs-rest, weighted. With weighted averaging, Recall is equal to Accuracy, so those two columns match.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7271 | 0.8728 | 0.7184 | 0.7271 | 0.7158 | 0.5528 |
| Decision Tree | 0.8112 | 0.8472 | 0.8112 | 0.8112 | 0.8112 | 0.6969 |
| kNN | 0.8316 | 0.9392 | 0.8299 | 0.8316 | 0.8298 | 0.7274 |
| Naive Bayes | 0.0936 | 0.6819 | 0.4322 | 0.0936 | 0.0576 | 0.0623 |
| Random Forest (Ensemble) | 0.8702 | 0.9633 | 0.8710 | 0.8702 | 0.8680 | 0.7893 |

### Graph Representation
The graph below shows all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for all 5 models.

![Metrics comparison](metrics_comparison.png)

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Lowest of the working models (accuracy 0.727, MCC 0.553). It draws only straight lines between classes, and this data is not separated that way. Its AUC (0.873) is still fair. |
| Decision Tree | Accuracy 0.811, much better than Logistic Regression. Its AUC (0.847) is lower, because one tree gives almost only 0 or 1 probabilities. One tree can also overfit. |
| kNN | Accuracy 0.832 and AUC 0.939, second best on every metric. Scaling helped it a lot. It is slow when predicting. |
| Naive Bayes | Very poor (accuracy 0.094, below random guessing of about 14%). It expects bell-shaped, independent features, but 44 of the 54 columns are only 0 or 1. It labelled about half of the test rows as class 7, which is only about 3.5% of the data, and it predicted class 2 (about 49% of the data) only 74 times. |
| Random Forest (Ensemble) | Best on all 6 metrics (accuracy 0.870, AUC 0.963, MCC 0.789). Many trees together reduce the overfitting of one tree. Most of its mistakes are between class 1 and class 2. |
| Overall Winner for your dataset? | Random Forest. It has the highest value on all 6 metrics. kNN is second but slower to predict. These results are from one train-test split. |
