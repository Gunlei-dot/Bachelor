import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import traceback
import shap
from mlflow.models import infer_signature
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # metrics for manual model eval
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


mlflow.sklearn.autolog(max_tuning_runs=1)
mlflow.set_experiment("new data testing")

tsv1 = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\data\labels.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly
tsv2 = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\data\prostate_stats.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly

patient_data = pd.merge(tsv1, tsv2, left_index=True, right_index=True)
patient_data = patient_data.dropna() # Drop rows with missing values for simplicity 
patient_data = patient_data.drop(columns=['pseudo_id', 'sex', 'pseudo_patid', 'pid', 'cx_px', 'cy_px', 'cz_px', 'cx', 'cy', 'cz']) # Drop patient_id as it's not a feature for prediction
patient_data = patient_data[patient_data.label != 2] # Remove rows with label 2 as these are not relevant for binary classification
patient_data = patient_data[patient_data.psa != 'NA'] # remove rows with no psa value till i find a better solution
patient_data = patient_data[patient_data.staging != 'primary'] # remove rows with primary staging till i find a better solution

patient_data['age'] = patient_data['age'].astype(float) # convert any ints to float to stop MLflow whining
patient_data['px'] = patient_data['px'].astype(float) 
patient_data['min'] = patient_data['min'].astype(float)
patient_data['max'] = patient_data['max'].astype(float)
patient_data['rmin'] = patient_data['rmax'].astype(float)
patient_data['mean'] = patient_data['mean'].astype(float)
patient_data['vol_pix'] = patient_data['vol_pix'].astype(float)
patient_data['vol_mm3'] = patient_data['vol_mm3'].astype(float)
patient_data['sd'] = patient_data['sd'].astype(float)

train_mask = patient_data['set'] == 'train' 
test_mask = patient_data['set'] == 'val' # splitting train/val before one-hot  

X = pd.get_dummies(patient_data.drop("label", axis=1)) # dummies for categorical variables since forest doesn't handle them directly
X.index = patient_data.index
y = patient_data[["label"]].astype(int) # Keeping y as DataFrame for easier handling of set indicators


X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

X_train = X_train.drop(columns=['set_train', 'set_val'], errors= 'ignore') # Drop the set indicator columns
X_test = X_test.drop(columns=['set_train', 'set_val'], errors = 'ignore')

y_test = np.array(y_test).astype(int) # Convert y_test to a NumPy array of strings
y_train = np.array(y_train).astype(int) # Convert y_train to a NumPy array of strings

y_test = y_test.squeeze()
y_train = y_train.squeeze() # sections sets up train test split and ensures proper data types

eval_data= X_test.copy()
eval_data['label']= y_test #create eval data for flow evaluation

param_grid = {
    'random_state': [42],
    'scale_pos_weight': [0.15, 0.2, 0.25, 0.3],
    'n_estimators': [300],
    'max_depth': [4],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'eval_metric': ['logloss'],
    'objective': ['binary:logistic'],
}

# Dropping the set indicator columns after the split
try:
    with mlflow.start_run(run_name='ND_param_tuning') as run: # Start an MLflow run to log parameters, metrics, and the model   
        print("MLflow run started successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        classifier = XGBClassifier()
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='recall', error_score='raise')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        model = XGBClassifier(**best_params)
        model = model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]  #implemtent probs for threshold tuning
        eval_data['pred_prob'] = probs

        thresholds = np.linspace(0.1, 0.5, 9)  # adjust for high-recall
        for t in thresholds:
            preds = (probs >= t).astype(int)
            recall = recall_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            print(f"Threshold {t:.2f} | Recall: {recall:.3f} | FN: {cm[1,0]}")
        
         # do manual parameter tracking so only the important bits are saved, reduce to the best estimator per run
    
        signature = infer_signature(X_train, model.predict(X_train)) 
        model_info = mlflow.sklearn.log_model(model, name="threshold_testing_boost", signature=signature)                ##Adjust type depending on model used
        mlflow.log_metric(f"recall_threshold_{t:.2f}", recall)
        
        
        
        result = mlflow.evaluate(
            model_info.model_uri,
            eval_data,
            targets= "label",
            model_type= "classifier",
            evaluators="default",
        )
        


    mlflow.end_run()  

except Exception as e:
    print("An error occurred during the MLflow run:")
    traceback.print_exc()
    mlflow.end_run()
    raise

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
print(f"Test score: {best_score:.3f}")
print(f"Recall: {result.metrics['recall_score']:.3f}")
print(f"F1 Score: {result.metrics['f1_score']:.3f}")
print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
print("Classification Report:\n", classification_report(y_test, preds))
