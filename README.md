Titanic Survival Prediction
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset used is the famous Titanic dataset, which contains various features about the passengers, such as age, gender, ticket class, and more. The goal is to build a model that accurately predicts whether a passenger survived the disaster.

Project Structure
data/: Directory containing the Titanic dataset.
notebooks/: Jupyter notebooks with data exploration, model training, and evaluation.
src/: Python scripts for data preprocessing, model training, and evaluation.
README.md: Project documentation.
Dataset
The dataset contains information on 891 passengers, with the following features:

Survived: Whether the passenger survived (1) or not (0).
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Passenger name.
Sex: Passenger gender.
Age: Passenger age.
SibSp: Number of siblings or spouses aboard the Titanic.
Parch: Number of parents or children aboard the Titanic.
Ticket: Ticket number.
Fare: Passenger fare.
Cabin: Cabin number.
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
Feature Engineering
To improve the model's performance, several new features were engineered:

Family Size: Combined SibSp and Parch to create a family_size feature, representing the total number of family members aboard, including the passenger.
Age Bins: The Age column was binned into categorical variables representing different age groups (e.g., Child, Teen, Adult).
Interaction Features: Created interaction features like Pclass_Fare (product of Pclass and Fare) and FamilySize_Pclass (product of family_size and Pclass).

Model Training
The model used for this project was Logistic Regression. The steps taken include:

Data Preprocessing:

Handled missing values by filling them with the median.
One-hot encoded categorical features like Sex, Embarked, and the newly created Title and AgeBin features.

Model Training:

The model was trained using the features described above.
The training set consisted of 80% of the data, and the remaining 20% was used as the test set.

Model Evaluation:

Accuracy: The model achieved an accuracy of approximately 80%.
Confusion Matrix: Provided a breakdown of true positives, true negatives, false positives, and false negatives.
Classification Report: Included precision, recall, and F1-scores for both classes (survived and did not survive).

Results
After adding new features, the model's performance was as follows:

Accuracy: 79.9%
Confusion Matrix:
[[90 15]
 [21 53]]
Classification Report:
markdown
Copy code
               precision    recall  f1-score   support

           0       0.81      0.86      0.83       105
           1       0.78      0.72      0.75        74

    accuracy                           0.80       179
   macro avg       0.80      0.79      0.79       179
weighted avg       0.80      0.80      0.80       179
While the accuracy slightly decreased from the baseline, the results indicate a balanced performance across both classes.

Future Work
To further improve the model, the following steps are suggested:

Feature Selection: Use techniques like Recursive Feature Elimination (RFE) or Lasso Regression to select the most important features.
Hyperparameter Tuning: Experiment with different hyperparameters to optimize the Logistic Regression model.
Model Exploration: Try more complex models like Random Forests or Gradient Boosting to capture non-linear relationships.
Cross-Validation: Implement cross-validation to ensure the model’s performance is consistent across different subsets of the data.
Conclusion
This project demonstrates the importance of feature engineering in building a predictive model. While adding new features did not drastically improve the performance, it provided valuable insights into the factors influencing survival on the Titanic.

License
This project is licensed under the MIT License.
