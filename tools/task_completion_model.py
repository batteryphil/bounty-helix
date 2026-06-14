import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TaskCompletionModel:
    """
    A machine learning model to predict and complete tasks based on past performance.
    """
    def __init__(self, task_data):
        self.task_data = task_data
        self.model = RandomForestClassifier()
        
    def preprocess_data(self):
        X = []
        y = []
        for task in self.task_data:
            X.append(task['context'])
            y.append(task['outcome'])
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict_outcome(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def evaluate_model(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def predict_task_outcome(self, task):
        context = task['context']
        context = np.array(context).reshape(1, -1)
        outcome = self.model.predict(context)
        return outcome[0]

def main():
    with open('task_data.json') as f:
        task_data = json.load(f)
        
    model = TaskCompletionModel(task_data)
    X_train, X_test, y_train, y_test = model.preprocess_data()
    model.train_model(X_train, y_train)
    y_pred = model.predict_outcome(X_test)
    accuracy = model.evaluate_model(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    new_task = {'context': [['task1', 'task2', 'task3']]}
    predicted_outcome = model.predict_task_outcome(new_task)
    print(f'Predicted outcome for new task: {predicted_outcome}')

if __name__ == '__main__':
    main()