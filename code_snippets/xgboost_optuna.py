import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# * empty dataframes for example, add test train split, this example is on an all data problem
X = pd.DataFrame()
y = pd.DataFrame()

def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic", # * change objective to match your problem
        # "tree_method": "hist",
        # defines booster, gblinear for linear functions.
        "booster": "gbtree",
        # "booster": "dart",
        # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85, step=0.009),
        # number of estimators
        # "n_estimators": trial.suggest_int("n_estimators", 500, 1000, step=100)
        "n_estimators": 1000
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 5, step=1)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 25, step=1)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        # param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    replace = {',':'_', '<':'LT', '>':'GT'}
    X_xgb = correct_feature_names(X, replace)
    xgb_c = xgb.XGBClassifier(**param)
    xgb_c.fit(X_xgb, y)
    pred = xgb_c.predict_proba(X_xgb)
    preds = pred[:, 1:].flatten()
#     pred_labels = np.rint(preds)
    roc = roc_auc_score(y, preds)
    return roc
#     f_score = f1_score(xgb.y_test, preds)
#     return f_score

def correct_feature_names(df: pd.DataFrame, replace: dict) -> pd.DataFrame:
    """_correct_feature_names _summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    for key, value in replace.items():
        df.columns = df.columns.str.replace(key, value)
    return df

# * Create study and optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5, timeout=600)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
parameters = trial.params

replace = {',':'_', '<':'LT', '>':'GT'}
X_xgb = correct_feature_names(X, replace)
xgb_c = xgb.XGBClassifier(**parameters)

xgb_c.fit(X_xgb, y)

preds = xgb_c.predict(X_xgb)

f_score = f1_score(y, preds)

c_report = classification_report(y, preds)
print(c_report)