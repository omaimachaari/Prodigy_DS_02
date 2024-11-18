# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv("c:/Users/lenovo/Downloads/train.csv")


# Dropping rows with missing 'Age' values
df = df.dropna(subset=['Age'])

# Renaming columns for better readability
df.rename(columns={'Pclass': 'Ticket_class', 'SibSp': 'nbr_of_siblings', 'Parch': 'nbr_of_parents'}, inplace=True)

# Dropping the 'Name' column since it's not useful for our analysis
df.drop('Name', axis=1, inplace=True)


# Creating 'AgeGroup' based on age bins
age_bins = [0, 12, 18, 40, 60, 100]
age_labels = ['Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Imputing missing values in 'Age' with the median
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])

# Creating a new feature 'FamilySize' by combining 'nbr_of_siblings' and 'nbr_of_parents'
df['FamilySize'] = df['nbr_of_siblings'] + df['nbr_of_parents'] + 1

# Creating 'FamilySizeCategory' based on family size bins
family_bins = [0, 2, 4, float('inf')]
family_labels = ['small', 'medium', 'large']
df['FamilySizeCategory'] = pd.cut(df['FamilySize'], bins=family_bins, labels=family_labels, right=False)

# Extracting the first letter of the 'Cabin' column to create 'Cabin_deck'
df['Cabin_deck'] = df['Cabin'].str[0].fillna('Unknown')

# Encoding categorical variables using LabelEncoder
label_encoder = LabelEncoder()
df['Cabin_deck'] = label_encoder.fit_transform(df['Cabin_deck'])
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'].fillna('Unknown'))


# Selecting features (including the newly created ones)
X = df[['Age', 'FamilySize', 'Cabin_deck', 'Sex', 'Ticket_class', 'Embarked']]
y = df['Survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Getting the best model and evaluating accuracy
best_rf_model = grid_search.best_estimator_
test_accuracy = best_rf_model.score(X_test, y_test)
print("Best parameters:", grid_search.best_params_)
print("Best score (Cross-Validation):", grid_search.best_score_)
print(f"Test Accuracy: {test_accuracy}")


# Convert categorical columns to numerical using one-hot encoding for correlation analysis
df_encoded = pd.get_dummies(df, drop_first=True)

# Plotting the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
