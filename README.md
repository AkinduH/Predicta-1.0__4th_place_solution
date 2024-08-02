# Predicta-1.0
Data Science Competition by IEEE - University of Peradeniya

![1](https://github.com/user-attachments/assets/128d2ad0-9f3b-442c-8d9b-58f552819c6f)

# Our approach- Team Data Nexus


## Time Series Problem (Question 1)

### Algorithm Selection:
For the time series prediction problem, three models were evaluated:

GradientBoostedTreesModel (using TensorFlow Decision Forests)

CatBoostRegressor

Prophet

### Model Architectures:

#### GradientBoostedTreesModel:

Task: Regression
Compiled with MSE metric
Hyperparameters: 200 trees, "BEST_FIRST_GLOBAL" growing strategy, max depth 25, L1 and L2 regularization at 0.01 each, minimum examples per node set to 2, learning rate at 0.1, subsampling at 0.8, and "RANDOM" algorithm for categorical features.

#### CatBoostRegressor:

Task: Regression
Hyperparameters: 1000 iterations, learning rate 0.04, depth 8, l2_leaf_reg 6, bagging temperature 0.8, early stopping rounds 50, verbose 100.

#### Prophet:

Task: Time series prediction
Utilized a moving average for seven days as a regressor and all data of average temperature.

### Performance Metrics:

GradientBoostedTreesModel: RMSE = 2.2816

CatBoostRegressor: RMSE = 2.0872

The Prophet model achieved superior performance compared to the other models, effectively capturing the overall trend of average temperature predictions​.

### Conclusion:

The Prophet model emerged as the most effective solution for the time series prediction task, demonstrating the lowest RMSE and accurately capturing the temperature trends. While the GradientBoostedTreesModel and CatBoostRegressor also performed well, the Prophet model's ability to integrate seasonal trends and external regressors made it the best choice for this particular dataset.




## Classification Problem (Question 2)

### Algorithm Selection:
For the classification problem, four models were evaluated:

Gradient Boosting Classifier

Support Vector Machine

XGBClassifier

LGBMClassifier

### Model Architectures and Training:

#### Gradient Boosting Classifier:
Achieved the highest accuracy on the test data (81.25%) but showed potential overfitting with perfect accuracy on the training data.

#### Support Vector Machine:
Showed the lowest performance, indicating a need for further tuning or feature engineering.

#### XGBClassifier:
Good performance with balanced accuracy on test and training data.

#### LGBMClassifier:
Slightly higher accuracy and precision compared to XGBClassifier, effectively handling the multiclass classification task.
### Performance Metrics:

GradientBoostingTreesModel: Accuracy on test data = 81.25%, Precision = 0.7958, Recall = 0.8125, F1-score = 0.7968

Support Vector Machine: Accuracy on test data = 58.33%, Precision = 0.6287, Recall = 0.5833, F1-score = 0.5917

XGBClassifier: Accuracy on test data = 75.00%, Precision = 0.7542, Recall = 0.7500, F1-score = 0.7431

LGBMClassifier: Accuracy on test data = 79.17%, Precision = 0.7907, Recall = 0.7917, F1-score = 0.7834​.

### Conclusion:
The Gradient Boosting Classifier achieved the highest accuracy for the classification task, but the potential overfitting suggests that further regularization or validation techniques might be necessary. The LGBMClassifier provided a strong balance of performance metrics, making it a reliable choice for practical deployment. Both the XGBClassifier and LGBMClassifier demonstrated robustness and effectiveness, while the Support Vector Machine requires further optimization to be competitive.

# Final Leaderboard

![1722317609723](https://github.com/user-attachments/assets/773a9afb-e990-4432-bbbb-3eada6f16470)

