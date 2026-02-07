import numpy as np
import matplotlib.pyplot as plt
from data_prepare import load_mnist_01



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss_and_grad(X, y, w, w0, lam):
    """
    X:  (n, d)
    y:  (n,)
    w:  (d,)
    w0: scalar
    """
    n, d = X.shape
    z = y * (X @ w + w0)          # (n,)

    log_terms = np.log1p(np.exp(-z))
    data_loss = log_terms.mean()
    reg_loss = lam / 2.0 * (np.sum(w ** 2) + w0 ** 2)
    loss = data_loss + reg_loss

    probs = sigmoid(-z)           # (n,)
    grad_w = -(y[:, None] * X * probs[:, None]).mean(axis=0) + lam * w
    grad_w0 = -(y * probs).mean() + lam * w0

    return loss, grad_w, grad_w0



def train_logistic_SGD(
    X, y, lam=0.1, lr=0.01, batch_size=128, num_iters=5000
):
    """
    Stochastic Gradient Descent for logistic regression.

    At each iteration we sample a mini-batch of size 'batch_size',
    compute the gradient on this batch, and update (w, w0).

    For plotting, we record the full training loss on the entire
    dataset after each parameter update.
    """
    n, d = X.shape

    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.01, size=d).astype(np.float32)
    w0 = np.float32(0.0)

    losses = []

    for t in range(num_iters):
        # sample a mini-batch
        idx = rng.choice(n, size=batch_size, replace=False)
        Xb = X[idx]
        yb = y[idx]

        # compute loss and grad on the mini-batch
        _, grad_w, grad_w0 = compute_loss_and_grad(Xb, yb, w, w0, lam)

        # SGD update
        lr_t = lr / np.sqrt(1 + t)
        w -= lr_t * grad_w
        w0 -= lr_t * grad_w0

        # evaluate full-batch training loss for logging
        loss_full, _, _ = compute_loss_and_grad(X, y, w, w0, lam)
        losses.append(loss_full)

        if (t + 1) % 500 == 0:
            print(f"[SGD] iter {t+1}/{num_iters}, full loss = {loss_full:.4f}")

    return w, w0, np.array(losses, dtype=np.float32)



def train_logistic_SGD_Nesterov(
    X, y, lam=0.1, lr=0.01, momentum=0.9, batch_size=128, num_iters=5000
):
    """
    Stochastic Gradient Descent with Nesterov's acceleration.

    We use a velocity term (v_w, v_w0) and compute the gradient at
    the look-ahead point (w + momentum * v_w, w0 + momentum * v_w0).
    """
    n, d = X.shape

    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.01, size=d).astype(np.float32)
    w0 = np.float32(0.0)

    v_w = np.zeros_like(w)
    v_w0 = np.float32(0.0)

    losses = []

    for t in range(num_iters):
        # sample mini-batch
        idx = rng.choice(n, size=batch_size, replace=False)
        Xb = X[idx]
        yb = y[idx]

        # look-ahead parameters
        w_look = w + momentum * v_w
        w0_look = w0 + momentum * v_w0

        # gradient at look-ahead point on the mini-batch
        _, grad_w, grad_w0 = compute_loss_and_grad(Xb, yb, w_look, w0_look, lam)

        # update velocities
        lr_t = lr / np.sqrt(1 + t)
        v_w = momentum * v_w - lr_t * grad_w
        v_w0 = momentum * v_w0 - lr_t * grad_w0

        # update parameters
        w = w + v_w
        w0 = w0 + v_w0

        # full-batch loss for logging
        loss_full, _, _ = compute_loss_and_grad(X, y, w, w0, lam)
        losses.append(loss_full)

        if (t + 1) % 500 == 0:
            print(f"[SGD+NAG] iter {t+1}/{num_iters}, full loss = {loss_full:.4f}")

    return w, w0, np.array(losses, dtype=np.float32)


def main():
    print("Loading MNIST 0/1 ...")
    X_train, y_train, X_val, y_val = load_mnist_01(subsample=True)
    print("Train shape:", X_train.shape, y_train.shape)

    lam = 0.1
    lr_sgd = 0.01
    batch_size = 128
    num_iters = 5000
    momentum = 0.9

    # (ii) SGD
    print("Training plain SGD ...")
    w_sgd, w0_sgd, losses_sgd = train_logistic_SGD(
        X_train, y_train,
        lam=lam, lr=lr_sgd,
        batch_size=batch_size,
        num_iters=num_iters
    )

    # (iii) SGD + Nesterov
    print("Training SGD with Nesterov acceleration ...")
    w_nag, w0_nag, losses_nag = train_logistic_SGD_Nesterov(
        X_train, y_train,
        lam=lam, lr=lr_sgd,
        momentum=momentum,
        batch_size=batch_size,
        num_iters=num_iters
    )

    iters = np.arange(num_iters)

    # plot both on semi-log scale
    plt.figure()
    plt.semilogy(iters, losses_sgd, label="SGD")
    plt.semilogy(iters, losses_nag, label="SGD + Nesterov")
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Training loss (log scale)")
    plt.title("SGD vs SGD + Nesterov on MNIST 0/1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgd_vs_nesterov.pdf")
    print("Saved figure to sgd_vs_nesterov.pdf")


if __name__ == "__main__":
    main()
