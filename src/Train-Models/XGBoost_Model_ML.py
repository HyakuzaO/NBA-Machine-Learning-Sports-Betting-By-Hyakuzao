import os
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score, roc_curve
from datetime_split import DateTimeSeriesSplitByDate

DATASET = "dataset_2021-25_new"
DB_PATH = "../../Data/dataset.sqlite"
MODEL_DIR = "../../Models"
MODEL_NAME = "XGBoost_{:.3f}AUC_ML-4.json"
STORAGE_URL = "sqlite:///../../Data/optuna_xgb.db"
STUDY_NAME = "xgb_nba_opt_multisoft"
N_TRIALS = 200
N_SPLITS = 8
TEST_SIZE_DATES = 0.10
GAP_DAYS = 1
EARLY_STOP = 200
MAX_ROUNDS = 5000
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.INFO)

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f'SELECT * FROM "{DATASET}"', con, index_col="index")
con.close()

y = df["Home-Team-Win"].astype(int)
d0 = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else pd.Series(pd.NaT, index=df.index)
d1 = pd.to_datetime(df["Date.1"], errors="coerce") if "Date.1" in df.columns else pd.Series(pd.NaT, index=df.index)
dates = d0.fillna(d1)

r = dates.rank(method="first")
r = (r - r.min()) / (r.max() - r.min() + 1e-12)
n_steps = 20
step_idx = np.floor(r * n_steps).astype(int).clip(0, n_steps)
scale = np.linspace(0.6, 2.2, n_steps + 1)
weights_all = scale[step_idx.values]

drop_cols = ['Score','Home-Team-Win','TEAM_NAME','Date','TEAM_NAME.1','Date.1','OU-Cover','OU']
X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
X = X_df.values.astype(float)

splitter = DateTimeSeriesSplitByDate(
    n_splits=N_SPLITS,
    date_column=None,
    test_size=TEST_SIZE_DATES,
    train_size=None,
    gap=GAP_DAYS,
    embargo=0,
    expanding=True,
    ensure_full_test=True,
)

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "multi:softprob",
        "num_class": 2,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 8.0),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
        "seed": SEED,
    }
    aucs = []
    for tr_idx, te_idx in splitter.split(X, dates=dates):
        x_tr, x_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        w_tr = weights_all[tr_idx]
        dtr = xgb.DMatrix(x_tr, label=y_tr, weight=w_tr)
        dte = xgb.DMatrix(x_te, label=y_te)
        cb = [xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True, maximize=False)]
        model = xgb.train(params, dtr, num_boost_round=MAX_ROUNDS, evals=[(dtr,"train"),(dte,"valid")], callbacks=cb, verbose_eval=False)
        p = model.predict(dte, iteration_range=(0, model.best_iteration + 1))[:, 1]
        aucs.append(roc_auc_score(y_te, p))
    mean_auc = float(np.mean(aucs))
    trial.report(mean_auc, step=0)
    return mean_auc

print(f"[OPTUNA] starting study {STUDY_NAME} with {N_TRIALS} trials")
study = optuna.create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print("[OPTUNA] done")
print(f"[OPTUNA] Best AUC: {study.best_value:.3f}")
print(f"[OPTUNA] Best params: {study.best_params}")

best_params = {
    "objective": "multi:softprob",
    "num_class": 2,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "eta": study.best_params["eta"],
    "max_depth": study.best_params["max_depth"],
    "min_child_weight": study.best_params["min_child_weight"],
    "subsample": study.best_params["subsample"],
    "colsample_bytree": study.best_params["colsample_bytree"],
    "reg_lambda": study.best_params["reg_lambda"],
    "reg_alpha": study.best_params["reg_alpha"],
    "gamma": study.best_params["gamma"],
    "max_bin": study.best_params["max_bin"],
    "seed": SEED,
}

acc_results = []
auc_results = []
best_auc = -1.0
best_model = None
best_auc_fname = None
best_te_pred_proba = None
best_te_true = None
best_te_pred = None

fold_id = 0
for tr_idx, te_idx in splitter.split(X, dates=dates):
    fold_id += 1
    x_train, x_test = X[tr_idx], X[te_idx]
    y_train, y_test = y.iloc[tr_idx].values, y.iloc[te_idx].values
    w_train = weights_all[tr_idx]
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    cb = [xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True, maximize=False)]
    model = xgb.train(best_params, dtrain, num_boost_round=MAX_ROUNDS, evals=[(dtrain,"train"),(dtest,"valid")], callbacks=cb, verbose_eval=False)
    preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    y_prob = preds[:, 1]
    y_hat = np.argmax(preds, axis=1)
    acc = round(accuracy_score(y_test, y_hat) * 100, 1)
    auc = roc_auc_score(y_test, y_prob)
    acc_results.append(acc)
    auc_results.append(auc)
    print(f"Fold {fold_id}: acc {acc}% | AUC {auc:.3f}")
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_te_pred_proba = y_prob.copy()
        best_te_true = y_test.copy()
        best_te_pred = y_hat.copy()


best_auc_fname = os.path.join(MODEL_DIR, MODEL_NAME.format(best_auc))
best_model.save_model(best_auc_fname)

acc_results = np.array(acc_results, dtype=float)
auc_results = np.array(auc_results, dtype=float)
print(f"CV mean acc: {acc_results.mean():.2f}% | std: {acc_results.std():.2f}%")
print(f"CV mean AUC: {auc_results.mean():.3f} | std: {auc_results.std():.3f}")
print(f"Best AUC saved model: {best_auc:.3f} -> {best_auc_fname}")

overall_acc = round(accuracy_score(best_te_true, best_te_pred) * 100, 2)
cm = confusion_matrix(best_te_true, best_te_pred, labels=[0,1])
report = classification_report(best_te_true, best_te_pred, target_names=["class_0","class_1"], digits=3)
print("=== Metrics for best-AUC fold @ threshold 0.5 ===")
print(f"Accuracy: {overall_acc}%")
print("Confusion matrix [rows=true, cols=pred]:")
print(cm)
print("Classification report:")
print(report)

p = best_te_pred_proba
y_true = best_te_true
fpr, tpr, thr = roc_curve(y_true, p)
j = tpr - fpr
thr_j = thr[np.argmax(j)]
y_hat_j = (p >= thr_j).astype(int)

print("=== Threshold tuning on best-AUC fold (Youden J) ===")
print(f"Threshold: {thr_j:.3f}")
print(f"Accuracy: {accuracy_score(y_true, y_hat_j)*100:.2f}% | F1: {f1_score(y_true, y_hat_j):.3f}")
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(y_true, y_hat_j, labels=[0,1]))
print("Classification report:")
print(classification_report(y_true, y_hat_j, target_names=['class_0','class_1'], digits=3))
