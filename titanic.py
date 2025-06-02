# 1. Imports
# Data imports
import pandas as pd
import numpy as np

# Machine learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = pd.read_csv("titanic.csv")
data.info()
print(data.isnull().sum())
# 2. Data Cleaning and Feature Engineering
def preprocess_Data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)

    fill_missing_ages(df)

    # Convert Gender
    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

    # Feature Engineering
    """
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    """
    return df

# 3. Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
            
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
    

data = preprocess_Data(data)
# 4. Create Freatures / Target Variables  (Make Flashcards)
X = data.drop(columns=["Survived"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# 5. ML Preproccessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Hyperparameter Tuning
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors":range(1,21),
        "metric" : ["euclidean", "manhattan", "minkowski"],
        "weights" : ["uniform","distance"]
    }
    
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


best_model = tune_model(X_train, y_train)
# 7. Predicating and Evaluate
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)
# 8. Plot

def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived","Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()
    
    
plot_model(matrix)   