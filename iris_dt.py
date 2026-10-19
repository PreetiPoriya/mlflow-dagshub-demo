import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='PreetiPoriya', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/PreetiPoriya/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# split the datasest into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

# Define the parameters for the random forest model
max_depth = 1

# apply mlflow

mlflow.set_experiment('iris-dt')

with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(x_train,y_train)
    y_pred = dt.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy',accuracy)

    # create a confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot = True, fmt='d', cmap='Blues', xticklables = iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('confusion matrix')

    # save the plot as an artifact
    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(dt,'decision_tree')

    mlflow.set_tag('author','preeti')
    mlflow.set_tag('model','decision_tree')
    print('accuracy',accuracy)