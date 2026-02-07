import torch
import matplotlib.pyplot as plt
from vae_model import Encoder, Decoder

def sample_from_vae(decoder, device="cpu", save_path="vae_random_samples.pdf"):

    decoder.eval()

    # 8 latent z
    z = torch.randn(8, 8).to(device)      # (8, latent_dim)
    y = decoder(z)                        # (8,196)
    imgs = y.view(-1, 14, 14).detach().cpu()

    # 8 images
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(f"z sample {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load("decoder_hw4_p1.pt", map_location=device))
    decoder.eval()

    sample_from_vae(decoder, device=device,
                    save_path="hw4_p1_random_samples.pdf")


if __name__ == "__main__":
    main()
