import numpy as np
import matplotlib.pyplot as plt
from data_prepare import load_mnist_01
from Gradient_descent import sigmoid , compute_loss_and_grad

def train_logistic_NAG(X, y, lam=0.1, lr=0.001, momentum=0.9, num_iters=5000):
    """
    Train logistic regression using Nesterov's accelerated gradient.

    Parameters
    ----------
    X : (n, d) numpy array
    y : (n,) numpy array with labels in {+1, -1}
    lam : float
        L2 regularization strength.
    lr : float
        Learning rate (step size).
    momentum : float
        Nesterov momentum parameter, typically in [0.75, 0.95].
    num_iters : int
        Number of parameter updates.

    Returns
    -------
    w : (d,) numpy array
        Final weight vector.
    w0 : float
        Final bias term.
    losses : (num_iters,) numpy array
        Training loss at each iteration.
    """
    n, d = X.shape

    # Initialize parameters (same seed as GD so that w, w0 start the same)
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.01, size=d).astype(np.float32)
    w0 = np.float32(0.0)

    # Initialize velocities for w and w0
    v_w = np.zeros_like(w)
    v_w0 = np.float32(0.0)

    losses = []

    for t in range(num_iters):
        # Look-ahead parameters (Nesterov step)
        w_look = w + momentum * v_w
        w0_look = w0 + momentum * v_w0

        # Compute loss and gradient at the look-ahead point
        loss, grad_w, grad_w0 = compute_loss_and_grad(X, y, w_look, w0_look, lam)

        if t % 200 == 0:
            print(f"[NAG] iter {t}, loss={loss:.4f}, ||grad_w||={np.linalg.norm(grad_w):.4e}, grad_w0={grad_w0:.4e}")

        losses.append(loss)

        # Update velocities
        v_w = momentum * v_w - lr * grad_w
        v_w0 = momentum * v_w0 - lr * grad_w0

        # Update parameters using the new velocity
        w = w + v_w
        w0 = w0 + v_w0

        if (t + 1) % 200 == 0:
            print(f"[NAG] iter {t+1}/{num_iters}, loss = {loss:.4f}")

    return w, w0, np.array(losses, dtype=np.float32)


# ---------- main: training + semi-log plots + slope ----------

def main():
    print("Loading MNIST 0/1 ...")
    X_train, y_train, X_val, y_val = load_mnist_01(subsample=True)
    print("Train shape:", X_train.shape, y_train.shape)

    lam = 0.1
    lr = 0.001
    momentum = 0.9
    num_iters = 5000

    print("Training with Nesterov accelerated gradient ...")
    w, w0, losses = train_logistic_NAG(
        X_train, y_train,
        lam=lam, lr=lr,
        momentum=momentum,
        num_iters=num_iters
    )

    iters = np.arange(len(losses))

    # 1. semi-log of loss
    plt.figure()
    plt.semilogy(iters, losses)
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Training loss (log scale)")
    plt.title("Logistic Regression with Nesterov AGD on MNIST 0/1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("nesterov_loss.pdf")
    print("Saved figure to nesterov_loss.pdf")

    # 2. semi-log of loss gap
    loss_arr = np.array(losses)
    loss_star = loss_arr.min()       # approximate ℓ(w*)
    gap = loss_arr - loss_star + 1e-4
    plt.figure()
    plt.semilogy(iters, gap)
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Training loss gap (log scale)")
    plt.title("Nesterov: log(loss gap)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("nesterov_loss_gap.pdf")
    print("Saved figure to nesterov_loss_gap.pdf")

    # 3. estimate slope and implied condition number (for Nesterov)
    burn_in = len(losses) // 4     # ignore early iterations
    slope, intercept = np.polyfit(iters[burn_in:], np.log(losses[burn_in:]), 1)
    print(f"[Nesterov] Slope of log(loss) vs iters (after burn-in) = {slope:.4e}")

    # For Nesterov, theory predicts slope ≈ -1 / sqrt(kappa)
    sqrt_kappa_est = -1.0 / slope
    kappa_est = sqrt_kappa_est ** 2
    print(f"[Nesterov] Estimated kappa from slope (≈ (-1/slope)^2) ≈ {kappa_est:.2f}")


if __name__ == "__main__":
    main()