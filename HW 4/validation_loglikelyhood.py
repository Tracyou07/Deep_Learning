import torch
import torch.nn as nn
import torch.optim as optim
from vae_model import Encoder ,Decoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import build_mnist_10k_dataset ,build_mnist_val_dataset



def reparameterize(mu, logvar, num_samples=2):

    batch_size, latent_dim = mu.shape
    std = torch.exp(0.5 * logvar)                # σ
    eps = torch.randn(num_samples, batch_size, latent_dim, device=mu.device)
    mu_expand = mu.unsqueeze(0)                  # (1, B, D)
    std_expand = std.unsqueeze(0)                # (1, B, D)
    z = mu_expand + std_expand * eps             # (S, B, D)
    return z


def vae_loss(x, mu, logvar, decoder, num_samples=2):
    batch_size, dim = x.shape
    # ---- 1. 采样 z ----
    z_samples = reparameterize(mu, logvar, num_samples=num_samples)  # (S,B,8)

    # ---- 2. reconstruction log-likelihood ----
    # x: (B,D) -> (1,B,D)
    x_expand = x.unsqueeze(0)  # (1,B,D)
    recon_log_probs = []

    for s in range(num_samples):
        z = z_samples[s]                      # (B,8)
        y = decoder(z)                        # (B,196), in (0,1)
        # Bernoulli log p(x|z) = sum_i x_i log y_i + (1-x_i) log(1-y_i)
        eps = 1e-8
        log_pxz = x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps)
        log_pxz = log_pxz.sum(dim=1)          # (B,)
        recon_log_probs.append(log_pxz)

    # recon_term
    recon_log_probs = torch.stack(recon_log_probs, dim=0)  # (S,B)
    recon_term = recon_log_probs.mean(dim=0)               # (B,)
    recon_term_mean = recon_term.mean()                    # scalar

    # KL = -0.5 * ∑_i (1 + log σ_i^2 - μ_i^2 - σ_i^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)  # (B,)
    kl_mean = kl.mean()

    elbo = recon_term - kl          # (B,)
    loss = -elbo.mean()             # 

    return loss, recon_term_mean.item(), kl_mean.item()


def compute_val_loglik(encoder, decoder, val_loader, device="cpu", num_samples=2):
    """
    从验证集中取 100 张图片，计算平均 E[log p(x|z)]
    """
    encoder.eval()
    decoder.eval()

    for x, _ in val_loader:
        x = x.view(x.size(0), -1).to(device).float()  # (B,196)
        mu, logvar = encoder(x)
        _, recon_avg, _ = vae_loss(x, mu, logvar, decoder, num_samples=num_samples)
        return recon_avg  

    return None  


def train_vae(train_loader, val_loader, device="cpu", lr=1e-3, num_epochs=10):
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    recon_history = []  
    kl_history = []      

    val_loglik_history = []  
    val_steps = []           

    encoder.train()
    decoder.train()

    global_step = 0  

    for epoch in range(num_epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(x.size(0), -1).to(device).float()

            mu, logvar = encoder(x)
            loss, recon_avg, kl_avg = vae_loss(
                x, mu, logvar, decoder, num_samples=2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            recon_history.append(recon_avg)
            kl_history.append(kl_avg)


            if global_step % 100 == 0:
                val_ll = compute_val_loglik(encoder, decoder, val_loader, device=device)
                val_loglik_history.append(val_ll)
                val_steps.append(global_step)
                print(f"[Val] step {global_step}, val log p(x|z) = {val_ll:.4f}")

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Step {batch_idx+1}, "
                      f"GlobalStep={global_step}, Loss={loss.item():.4f}, "
                      f"Recon={recon_avg:.4f}, KL={kl_avg:.4f}")

    return encoder, decoder, recon_history, kl_history, val_steps, val_loglik_history


def plot_train_val_recon(recon_history, val_steps, val_loglik_history,
                         save_path="train_val_recon.pdf"):
    steps = range(1, len(recon_history) + 1)

    plt.figure(figsize=(8,5))

    plt.plot(steps, recon_history, label="Train reconstruction term")

    plt.scatter(val_steps, val_loglik_history, label="Validation log-likelihood",
                marker="o")

    plt.xlabel("Number of weight updates")
    plt.ylabel("E[log p(x|z)]")
    plt.title("Train Reconstruction Term vs Validation Log-likelihood")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    mnist_10k = build_mnist_10k_dataset(root="./data")
    mnist_val = build_mnist_val_dataset(root="./data")

    val_loader = DataLoader(
        mnist_val,
        batch_size=100,   
        shuffle=False
    )

    batch_size = 128
    train_loader = DataLoader(
        mnist_10k,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    num_epochs = 30
    lr = 1e-3
    encoder, decoder, recon_history, kl_history, val_steps, val_loglik_history = train_vae(
        train_loader,
        val_loader,
        device=device,
        lr=lr,
        num_epochs=num_epochs
    )

    plot_train_val_recon(recon_history, val_steps, val_loglik_history,
                        save_path="hw4_p1_train_val_recon.pdf")


if __name__ == "__main__":
    main()
