import torch
import matplotlib.pyplot as plt
from vae_model import Encoder, Decoder
from dataloader import build_mnist_10k_dataset

def plot_8_reconstructions(encoder, decoder, dataset, device="cpu",
                           save_path="reconstruction_8.pdf"):

    encoder.eval()
    decoder.eval()

    # 8 images
    idxs = torch.randint(0, len(dataset), (8,))
    
    originals = []
    recons = []

    for idx in idxs:
        x, _ = dataset[idx]            # x: (1,14,14) tensor
        x_flat = x.view(1, -1).to(device).float()  

        # ---- Encoder ----
        mu, logvar = encoder(x_flat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps             # reparameterization

        # ---- Decoder ----
        y = decoder(z)                 # (1,196)
        y_img = y.view(14, 14).detach().cpu()

        originals.append(x.view(14,14).cpu())
        recons.append(y_img)

    fig, axes = plt.subplots(8, 2, figsize=(4, 16))

    for i in range(8):
        axes[i, 0].imshow(originals[i], cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recons[i], cmap="gray")
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"8 image sved at {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    encoder.load_state_dict(torch.load("encoder_hw4_p1.pt", map_location=device))
    decoder.load_state_dict(torch.load("decoder_hw4_p1.pt", map_location=device))

    encoder.eval()
    decoder.eval()


    mnist_10k = build_mnist_10k_dataset("./data")

    plot_8_reconstructions(
        encoder, decoder, mnist_10k, device=device,
        save_path="hw4_reconstruction_8.pdf"
    )




if __name__ == "__main__":
    main()
