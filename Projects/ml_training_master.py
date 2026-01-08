import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import traceback
from mlflow.models import infer_signature
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay  # metrics for manual model eval

tsv1 = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\data\labels.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly
tsv2 = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\data\prostate_stats.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly

patient_data = pd.merge(tsv1, tsv2, left_index=True, right_index=True)
patient_data = patient_data.dropna() # Drop rows with missing values for simplicity 
patient_data = patient_data.drop(columns=['pseudo_id', 'sex', 'pseudo_patid', 'pid', 'cx_px', 'cy_px', 'cz_px', 'cx', 'cy', 'cz']) # Drop patient_id as it's not a feature for prediction
patient_data = patient_data[patient_data.label != 2] # Remove rows with label 2 as these are not relevant for binary classification
patient_data = patient_data[patient_data.psa != 'NA'] # remove rows with no psa value till i find a better solution
patient_data = patient_data[patient_data.staging != 'primary'] # remove rows with primary staging till i find a better solution

patient_data['age'] = patient_data['age'].astype(float) # convert psa to float
patient_data['px'] = patient_data['px'].astype(float) # convert psa to float

train_mask = patient_data['set'] == 'train' 
test_mask = patient_data['set'] == 'val' # splitting train/val before one-hot encoding 

X = pd.get_dummies(patient_data.drop("label", axis=1)) # dummies for categorical variables since forest doesn't handle them directly
X.index = patient_data.index # keep original indices for proper splitting later
y = patient_data["label"].astype(int) # Keeping y as DataFrame for easier handling of set indicators

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train shape: {X_train.shape} {y_train.shape}") #double check proper set splits

# Dropping the set indicator columns after the split
X_train = X_train.drop(columns=['set_train', 'set_val'], errors= 'ignore') # Drop the set indicator columns
X_test = X_test.drop(columns=['set_train', 'set_val'], errors = 'ignore') # Drop the set indicator columns

y_test = np.array(y_test).astype(int) # Convert y_test to a NumPy array of strings
y_train = np.array(y_train).astype(int) # Convert y_train to a NumPy array of strings

y_test = y_test.squeeze()
y_train = y_train.squeeze()

#ensure X and y have matching lengths
if len(X_train) != len(y_train):
    print("Mismatch between X_train and y_train lengths!")
    raise SystemExit()
if len(X_test) != len(y_test):
    print("Mismatch between X_test and y_test lengths!")
    raise SystemExit()

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Unique labels:", np.unique(y_train))

mlflow.set_experiment("Inital tests new data")

try:
    with mlflow.start_run() as run:  # Everything inside this block is logged
        print("MLflow run started successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")   
        
        params = {
            'random_state' : 42,
            'min_weight_fraction_leaf': 0.1,
        }

        # Log parameters
        mlflow.log_params(params)

        # Train model(chose whichever applies and set params accordingly)
       
        model = tree.DecisionTreeClassifier(**params)
        #model = RandomForestClassifier(**params)
        #model = GradientBoostingClassifier(**params)

        model = model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6,6))
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot(ax=ax, cmap="Greens")

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        # Log metrics and model
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_figure(fig, "confusion_matrix.png")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="ND_tree",             # Change model name accordingly
            signature=signature
        )

    mlflow.end_run()

except Exception as e:
    print("An error occurred during the MLflow run:")
    traceback.print_exc()
    mlflow.end_run()
    raise

print("Metrics logged to MLflow:")
print("Classification Report:\n", classification_report(y_test, y_pred))
