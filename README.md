#Titanic Data Analysis and Prediction with Machine Learning#
Project Overview
This project focuses on data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning model training using the popular Titanic dataset from Kaggle. The goal is to predict passenger survival based on various features like age, gender, ticket class, and family size.

Table of Contents
Project Overview
Dataset Information
Project Structure
Technologies Used
Setup and Installation
Exploratory Data Analysis (EDA)
Feature Engineering
Model Training and Evaluation
Results
Conclusion
References
Dataset Information
The dataset used in this project is the Titanic dataset provided by Kaggle. It includes information about the passengers aboard the Titanic, such as:

PassengerId: Unique identifier for each passenger.
Survived: Survival status (0 = No, 1 = Yes).
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Passenger's name.
Sex: Gender of the passenger.
Age: Age of the passenger.
SibSp: Number of siblings/spouses aboard.
Parch: Number of parents/children aboard.
Ticket: Ticket number.
Fare: Ticket fare.
Cabin: Cabin number.
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
Project Structure

├── hello_world.py           # Main Python script for data analysis and modeling
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── data/
    └── train.csv            # Titanic training dataset
Technologies Used
Python: Core programming language.
Pandas: Data manipulation and analysis.
NumPy: Numerical computing.
Scikit-Learn: Machine learning library.
Matplotlib & Seaborn: Data visualization.
GridSearchCV: Hyperparameter tuning.
Setup and Installation
Prerequisites
Ensure you have Python 3.8 or higher installed. You can check your Python version using:


python --version
Installation
Clone the repository:


git clone https://github.com/your-username/titanic-eda-ml.git
cd titanic-eda-ml
Install dependencies:


pip install -r requirements.txt
Download the Titanic dataset from Kaggle and place it in the data folder.

Exploratory Data Analysis (EDA)
Data Cleaning: Handling missing values in the 'Age' and 'Embarked' columns.
Feature Engineering:
Created new features like FamilySize, AgeGroup, and Cabin_deck.
Encoded categorical variables like Sex, Embarked, and Cabin_deck using LabelEncoder.
Visualization:
Visualized correlations between features using a heatmap.
Analyzed survival rates based on different groups (e.g., gender, ticket class, family size).
Feature Engineering
Age Grouping: Categorized passengers into groups (Child, Teenager, Adult, etc.).
Family Size: Added a new feature based on the total number of family members aboard.
Cabin Deck Extraction: Extracted the first letter of the cabin to represent the deck level.
Label Encoding: Converted categorical variables into numerical values.
Model Training and Evaluation
Model: Used a RandomForestClassifier for prediction.
Hyperparameter Tuning: Performed using GridSearchCV to find the best model parameters.
Metrics:
Accuracy Score: Evaluated the model's performance on the test dataset.
Best Parameters: Tuned parameters for optimal model performance.
Key Sections of the Code

# Loading the dataset
df = pd.read_csv("data/train.csv")

# Data Cleaning
df = df.dropna(subset=['Age'])

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Cabin_deck'] = df['Cabin'].str[0].fillna('Unknown')
df['Cabin_deck'] = LabelEncoder().fit_transform(df['Cabin_deck'])

# Model Training
X = df[['Age', 'FamilySize', 'Cabin_deck', 'Sex']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
Results
Best Hyperparameters:
n_estimators: 200
max_depth: None
min_samples_split: 2
min_samples_leaf: 1
Model Accuracy: The model achieved a test accuracy of 85% on the unseen test dataset.
Conclusion
This project demonstrates the process of performing data cleaning, exploratory data analysis, feature engineering, and machine learning model training on a real-world dataset. The RandomForest model provided good performance in predicting the survival of Titanic passengers based on the given features.

References
Kaggle Titanic Dataset
Scikit-Learn Documentation
Pandas Documentation
Seaborn Documentation
