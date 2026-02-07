from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# gabor fliter bank
def gabor_features_batch(x):
    # thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    # freqs  = [0.05, 0.20, 0.35]
    # bws    = [0.3, 0.6, 0.9]

    thetas = np.linspace(0, np.pi, 8, endpoint=False).tolist()                
    freqs  = [0.05, 0.12, 0.20, 0.28, 0.35, 0.42]                             
    bws = [3.0, 4.0, 5.0] 

    X = np.asarray(x, dtype=np.float32)
    N = X.shape[0]
    imgs = X.reshape(N, 14, 14)

    feats = []

    for th in thetas:
        for fr in freqs:
            for bw in bws:
                kernel = cv2.getGaborKernel((14, 14), sigma=4.0, theta=th,
                                            lambd=1.0/fr, gamma=0.5, psi=0,
                                            ktype=cv2.CV_32F)
                resp_list = []
                for im in imgs:
                    resp = cv2.filter2D(im, cv2.CV_32F, kernel)
                    resp_list.append(resp.ravel())
                feats.append(np.stack(resp_list, axis=0))  

    return np.concatenate(feats, axis=1).astype(np.float32)  # (N,7056)

# dataset MINIST
ds = fetch_openml('mnist_784', as_frame=False)

# Dtest/Dtrain
x_train, x_test, y_train, y_test = train_test_split(
    ds.data, ds.target, test_size=0.2, random_state=42
)
#sub-Dtest/Dtrain
x_train_sub, y_train_sub = sample_per_class(x_train, y_train, per_class=100)
x_test_sub, y_test_sub = sample_per_class(x_test, y_test, per_class=100)

x_train_sub = downsample_batch(x_train_sub)   
x_test_sub = downsample_batch(x_test_sub)

# Gabor feature generation
x_train_gabor = gabor_features_batch(x_train_sub)    
x_test_gabor = gabor_features_batch(x_test_sub)

print("Gabor feature shape:", x_train_gabor.shape)

#standarlization
scaler = StandardScaler(with_mean=True)        
x_train_std = scaler.fit_transform(x_train_gabor)
x_test_std  = scaler.transform(x_test_gabor )

#pca
pca = PCA(n_components=999, svd_solver='full', whiten=False, random_state=42)
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

#fit svm by subDtrain
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
grid.fit(x_train_pca, y_train_sub)

print("\n=== 5-fold CV mean accuracy for each C ===")
for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                             grid.cv_results_["std_test_score"],
                             grid.cv_results_["params"]):
    print(f"{params} -> {mean:.4f} ± {std:.4f}")

print("\nBest params (by 5-fold CV):", grid.best_params_)
print("Best CV mean accuracy:", grid.best_score_)

clf = grid.best_estimator_

y_pred = clf.predict(x_test_pca)
acc = accuracy_score(y_test_sub, y_pred)
sv_ratio = len(clf.support_) / len(x_train_gabor) 

print(f"Test accuracy = {acc:.4f}")
print(f"Support vector ratio: {sv_ratio:.4f}  ({len(clf.support_)} / {len(x_train_gabor)})")

