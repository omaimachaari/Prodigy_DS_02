import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("c:/Users/lenovo/Downloads/train.csv")

df = df.dropna(subset=['Age'])

df.rename(columns={'Pclass': 'Ticket_class', 'SibSp': 'nbr_of_siblings', 'Parch': 'nbr_of_parents'}, inplace=True)
df.drop('Name', axis=1, inplace=True)


bins = [0, 12, 18, 40, 60, 100]  
labels = ['Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])


df['FamilySize'] = df['nbr_of_siblings'] + df['nbr_of_parents'] + 1
bins = [0, 2, 4, float('inf')]
labels = ['small', 'medium', 'large']
df['FamilySizeCategory'] = pd.cut(df['FamilySize'], bins=bins, labels=labels, right=False)


df['Cabin_deck'] = df['Cabin'].str[0] 
df['Cabin_deck'] = df['Cabin_deck'].fillna('Unknown')
label_encoder = LabelEncoder()
df['Cabin_deck'] = label_encoder.fit_transform(df['Cabin_deck'])


X = df[['Age', 'FamilySize', 'Cabin_deck']]
y = df['Survived']  

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

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_rf_model = grid_search.best_estimator_

test_accuracy = best_rf_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")