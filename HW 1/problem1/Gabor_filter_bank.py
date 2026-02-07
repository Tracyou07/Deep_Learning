import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel

thetas = np.arange(0, np.pi, np.pi/4)        # 4
freqs  = np.arange(0.05, 0.5, 0.15)          # 3
bws    = np.arange(0.3, 1.0, 0.3)            # 3

def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    a, b = arr.min(), arr.max()
    return (arr - a) / (b - a + 1e-8)

for bw in bws:
    fig, axes = plt.subplots(len(thetas), len(freqs), figsize=(10, 8))
    for i, th in enumerate(thetas):
        for j, fr in enumerate(freqs):
            ker = gabor_kernel(frequency=fr, theta=th, bandwidth=bw)
            k_real = np.real(ker)
            ax = axes[i, j]
            ax.imshow(normalize_for_display(k_real), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"θ={np.degrees(th):.0f}°, f={fr:.2f}\nBW={bw:.1f}", fontsize=9)
    fig.suptitle(f"Gabor kernels (real part) — bandwidth={bw:.1f}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
