import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle as pickle
import numpy as np


def compare_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2%}")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print("\nBest performing model:", best_model[0])
    print("="*50 + "\n")
    
    return models[best_model[0]]


def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # Compare different models
  best_model = compare_models(X_train, X_test, y_train, y_test)
  
  # Train the best model again for final evaluation
  best_model.fit(X_train, y_train)
  y_pred = best_model.predict(X_test)
  
  print("\nFinal Model Performance:")
  print("-"*50)
  print(classification_report(y_test, y_pred))
  
  return best_model, scaler


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def main():
  data = get_clean_data()

  model, scaler = create_model(data)

  with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()