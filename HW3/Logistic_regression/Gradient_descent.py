import numpy as np

import matplotlib.pyplot as plt
from data_prepare import load_mnist_01


# loss and gradient
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss_and_grad(X, y, w, w0, lam):
    """
    X: (n, d)
    y: (n,)
    w: (d,)
    w0: scalar
    """
    n, d = X.shape
    # z_i = y_i (w^T x_i + w0)
    z = y * (X @ w + w0)          # (n,)

    # loss
    log_terms = np.log1p(np.exp(-z))
    data_loss = log_terms.mean()
    reg_loss = lam / 2.0 * (np.sum(w ** 2) + w0 ** 2)
    loss = data_loss + reg_loss

    # gradient
    probs = sigmoid(-z)           # (n,)
    grad_w = -(y[:, None] * X * probs[:, None]).mean(axis=0) + lam * w
    grad_w0 = -(y * probs).mean() + lam * w0

    return loss, grad_w, grad_w0


# GD
def train_logistic_GD(X, y, lam=0.1, lr=0.1, num_iters=2000):
    n, d = X.shape
    #initial same with (d) Nesterov 
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.01, size=d).astype(np.float32)
    w0 = np.float32(0.0)

    losses = []

    for t in range(num_iters):
        loss, grad_w, grad_w0 = compute_loss_and_grad(X, y, w, w0, lam)

        if t % 200 == 0:
             print(f"iter {t}, loss={loss:.4f}, ||grad_w||={np.linalg.norm(grad_w):.4e}, grad_w0={grad_w0:.4e}")

        losses.append(loss)

        w -= lr * grad_w
        w0 -= lr * grad_w0

        if (t + 1) % 200 == 0:
            print(f"iter {t+1}/{num_iters}, loss = {loss:.4f}")

    return w, w0, np.array(losses, dtype=np.float32)


def main():

    print("Loading MNIST 0/1 ...")
    X_train, y_train, X_val, y_val = load_mnist_01(subsample=True)
    print("Train shape:", X_train.shape, y_train.shape)

    lam = 0.1
    lr = 0.001     
    num_iters = 5000

    print("Training GD ...")
    w, w0, losses = train_logistic_GD(X_train, y_train, lam=lam, lr=lr, num_iters=num_iters)

    # 3. semi-log Y
    iters = np.arange(len(losses))
    plt.figure()
    plt.semilogy(iters, losses)
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Training loss (log scale)")
    plt.title("Logistic Regression with GD on MNIST 0/1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gd_loss.pdf")
    print("Saved figure to gd_loss.pdf")

    loss_arr = np.array(losses)
    loss_star = loss_arr.min()   # ℓ(w*)
    gap = loss_arr - loss_star + 1e-4  
    plt.semilogy(iters, gap)
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Training loss gap (log scale)")
    plt.title("log(loss gap)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gd_loss_gap.pdf")



    burn_in = len(losses) // 4     
    slope, intercept = np.polyfit(iters[burn_in:], np.log(losses[burn_in:]), 1)
    print(f"Slope of log(loss) vs iters (after burn-in) = {slope:.4e}")

    kappa_est = -1.0 / slope
    print(f"Estimated kappa (≈ -1/slope) ≈ {kappa_est:.2f}")


if __name__ == "__main__":
    main()
