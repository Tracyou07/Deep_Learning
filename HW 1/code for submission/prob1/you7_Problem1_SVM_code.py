from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
  
#down-sampler
def resize_14x14(im28):
        # type transform for cv2
        im28 = im28.astype(np.float32)
        return cv2.resize(im28, (14,14), interpolation=cv2.INTER_AREA)
def downsample_batch(X_flat):
    n = X_flat.shape[0]
    imgs28 = X_flat.reshape(n, 28, 28)
    imgs14 = np.stack([resize_14x14(im) for im in imgs28], axis=0)
    return imgs14.reshape(n, 14*14).astype(np.float32)

# sub-sample dataset creator
def sample_per_class(x, y, per_class=1000, random_state=42):
    rs = np.random.RandomState(random_state)
    keep_idx = []
    for digit in map(str, range(10)):
        idx = np.where(y == digit)[0]
        sel = rs.choice(idx, size=per_class, replace=False)
        keep_idx.append(sel)
    keep_idx = np.concatenate(keep_idx)
    return x[keep_idx], y[keep_idx]

# dataset MINIST
ds = fetch_openml('mnist_784', as_frame=False)

# Dtest/Dtrain
x_train, x_test, y_train, y_test = train_test_split(
    ds.data, ds.target, test_size=0.2, random_state=42
)
#sub-Dtest/Dtrain
x_train_sub, y_train_sub = sample_per_class(x_train, y_train, per_class=1000)
x_test_sub, y_test_sub = sample_per_class(x_test, y_test, per_class=1000)

x_train_sub = downsample_batch(x_train_sub)   
x_test_sub = downsample_batch(x_test_sub) 

x_test = downsample_batch(x_test)

#fit svm by gridsearchcv to find best para.c
param_grid = {"C": [0.1, 0.5, 1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]}  
base = svm.SVC(kernel='rbf', gamma='scale')

grid = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,       
    n_jobs=-1,
    refit=True,   
    verbose=3
)
grid.fit(x_train_sub, y_train_sub)

print("\n=== 5-fold CV mean accuracy for each C ===")
for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                             grid.cv_results_["std_test_score"],
                             grid.cv_results_["params"]):
    print(f"{params} -> {mean:.4f} ± {std:.4f}")

print("\nBest params (by 5-fold CV):", grid.best_params_)
print("Best CV mean accuracy:", grid.best_score_)

# fit the model by best c
clf = grid.best_estimator_
print("\n#SV:", len(clf.support_), " / ", len(x_train_sub),
      " (ratio = {:.4f})".format(len(clf.support_) / len(x_train_sub)))

# sub test vlidation
y_test_sub_pred = clf.predict(x_test_sub)
val_acc = accuracy_score(y_test_sub, y_test_sub_pred)
val_err = 1 - val_acc
cm_sub = confusion_matrix(y_test_sub, y_test_sub_pred, labels=list(map(str, range(10))))
print("\n[Sub-Set] Validation error: {:.4f} (acc={:.4f})".format(val_err, val_acc))

# test validation
y_test_pred = clf.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_err = 1 - test_acc
cm = confusion_matrix(y_test, y_test_pred, labels=list(map(str, range(10))))
print("\n[Test] Test error: {:.4f} (acc={:.4f})".format(test_err, test_acc))


