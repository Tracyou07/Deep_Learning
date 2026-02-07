import torch
import torch.nn as nn
import torch.optim as optim
from vae_model import Encoder ,Decoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import build_mnist_10k_dataset


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
    # sampling
    z_samples = reparameterize(mu, logvar, num_samples=num_samples)  # (S,B,8)

    # -log-likelihood 
    x_expand = x.unsqueeze(0)  # (1,B,D)
    recon_log_probs = []

    for s in range(num_samples):
        z = z_samples[s]                      # (B,8)
        y = decoder(z)                        # (B,196), in (0,1)
        eps = 1e-8
        log_pxz = x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps)
        log_pxz = log_pxz.sum(dim=1)          # (B,)
        recon_log_probs.append(log_pxz)

    # recon_term
    recon_log_probs = torch.stack(recon_log_probs, dim=0)  # (S,B)
    recon_term = recon_log_probs.mean(dim=0)               # (B,)
    recon_term_mean = recon_term.mean()                    # scalar

    #  KL(q(z|x)||N(0,I))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)  # (B,)
    kl_mean = kl.mean()

    # ELBO 
    elbo = recon_term - kl          # (B,)
    loss = -elbo.mean()            

    return loss, recon_term_mean.item(), kl_mean.item()


def train_vae(train_loader, device="cpu", lr=1e-3, num_epochs=30):
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    recon_history = []   
    kl_history = []      

    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            # x: (B,1,14,14)  -> (B,196)
            x = x.view(x.size(0), -1).to(device).float()

            mu, logvar = encoder(x)
            loss, recon_avg, kl_avg = vae_loss(
                x, mu, logvar, decoder, num_samples=2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_history.append(recon_avg)
            kl_history.append(kl_avg)

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Step {batch_idx+1}, "
                      f"Loss={loss.item():.4f}, Recon={recon_avg:.4f}, KL={kl_avg:.4f}")

    return encoder, decoder, recon_history, kl_history


def save_elbo_two_plots(recon_history, kl_history,
                        recon_path="recon_term.pdf",
                        kl_path="kl_term.pdf"):

    steps = range(1, len(recon_history) + 1)


    plt.figure(figsize=(7,5))
    plt.plot(steps, recon_history, label="Reconstruction term")
    plt.xlabel("Number of weight updates")
    plt.ylabel("E[log p(x|z)]")
    plt.title("Reconstruction Term vs Weight Updates")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(recon_path)
    plt.close()
    print(f"重建项图已保存到: {recon_path}")

    plt.figure(figsize=(7,5))
    plt.plot(steps, kl_history, label="KL term", color='orange')
    plt.xlabel("Number of weight updates")
    plt.ylabel("KL(q(z|x)||N(0,I))")
    plt.title("KL Term vs Weight Updates")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(kl_path)
    plt.close()
    print(f"KL 项图已保存到: {kl_path}")


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    mnist_10k = build_mnist_10k_dataset(root="./data")

    batch_size = 128
    train_loader = DataLoader(
        mnist_10k,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )


    num_epochs = 30
    lr = 1e-3
    encoder, decoder, recon_history, kl_history = train_vae(
        train_loader,
        device=device,
        lr=lr,
        num_epochs=num_epochs
    )

    torch.save(encoder.state_dict(), "encoder_hw4_p1.pt")
    torch.save(decoder.state_dict(), "decoder_hw4_p1.pt")
    print("Training finished, models saved to encoder_hw4_p1.pt / decoder_hw4_p1.pt")

 
    save_elbo_two_plots(recon_history, kl_history)

if __name__ == "__main__":
    main()
