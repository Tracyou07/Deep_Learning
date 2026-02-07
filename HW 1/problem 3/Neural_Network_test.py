import torchvision as thv
import numpy as np
import matplotlib.pyplot as plt

class embedding_t:
    """
    in:  h^l  (B,28,28)
    out: h^{l+1} (B,392) == flatten of (B,7,7,8)
    params: W (4,4,8), b (8,)
    """

    def __init__(self, seed=0):
        rs = np.random.RandomState(seed)
        # gaussian initialization
        self.w = rs.randn(4, 4, 8)
        self.b = rs.randn(8)

        # standarlize
        wnorm = np.linalg.norm(self.w.ravel())
        bnorm = np.linalg.norm(self.b)
        if wnorm > 0: self.w /= wnorm
        if bnorm > 0: self.b /= bnorm

        # gradient and cache
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.hl = None          # cache x
        self._patches = None    # cache patches
    
    # image <-> patch 
    def _im2patch(self, x):
        """
        x: (B,28,28) -> (B,7,7,4,4)
        """
        B = x.shape[0]
        x_reshaped = x.reshape(B, 7, 4, 7, 4)
        return x_reshaped.transpose(0, 1, 3, 2, 4)

    def _patch2im(self, patches):
        """
        (B,7,7,4,4) -> (B,28,28)
        """
        B = patches.shape[0]
        x_reshaped = patches.transpose(0, 1, 3, 2, 4)
        return x_reshaped.reshape(B, 28, 28)

    # forward
    def forward(self, h_l):
        """
        h_l: (B,28,28)
        return: h_{l+1}: (B,392)
        """
        assert h_l.ndim == 3 and h_l.shape[1:] == (28, 28)
        self.hl = h_l

        patches = self._im2patch(h_l)          
        self._patches = patches               

        h = np.einsum('bijuv, uvo -> bijo', patches, self.w)  
        h += self.b                                            

        return h.reshape(h.shape[0], -1)       

    # backward
    def backward(self, dh_next):
        # dim match
        if dh_next.ndim == 2:
            B = dh_next.shape[0]
            dh = dh_next.reshape(B, 7, 7, 8)
        else:
            dh = dh_next
            B = dh.shape[0]

        patches = self._patches  

        # gradient for b
        self.db = dh.sum(axis=(0, 1, 2))                         

        # gradient for w
        self.dw = np.einsum('bijuv, bijo -> uvo', patches, dh)   

        # gradient of patches for recursion chain rule 
        dpatches = np.einsum('bijo, uvo -> bijuv', dh, self.w)   
        dh_l = self._patch2im(dpatches)                          

        self._patches = None
        return dh_l

    def zero_grad(self):
        self.dw[...] = 0.0
        self.db[...] = 0.0

class linear_t:
    def __init__(self, in_dim=392, out_dim=10, seed=1):
        rs = np.random.RandomState(seed)
        self.w = rs.randn(out_dim, in_dim)
        self.b = rs.randn(out_dim)
        self.w /= max(np.linalg.norm(self.w), 1e-12)
        self.b /= max(np.linalg.norm(self.b), 1e-12)
        self.dw = np.zeros_like(self.w); self.db = np.zeros_like(self.b)
        self.hl = None

    def forward(self, h_l):  
        self.hl = h_l
        return h_l @ self.w.T + self.b

    def backward(self, dh_next):  
        self.dw = dh_next.T @ self.hl          
        self.db = dh_next.sum(axis=0)          
        dh_l = dh_next @ self.w                
        return dh_l

    def zero_grad(self):
        self.dw[...] = 0.0; self.db[...] = 0.0

class relu_t:
    def __init__(self):
        self.mask = None  

    def forward(self, h_l):
        self.mask = (h_l > 0)
        return np.where(self.mask, h_l, 0.0)

    def backward(self, dh_next):
        # upstream grad * 1{x>0}
        return dh_next * self.mask.astype(dh_next.dtype)

class softmax_ce_loss:
    def __init__(self):
        self.probs = None  
        self.y = None

    def forward(self, logits, y):
        """
        logits: (B, C), y: int labels (B,)
        return: scalar loss (float)
        """
        self.y = y
        z = logits - logits.max(axis=1, keepdims=True)  
        expz = np.exp(z)
        self.probs = expz / expz.sum(axis=1, keepdims=True)  

        B = logits.shape[0]
        loss = -np.log(self.probs[np.arange(B), y] + 1e-12).mean()
        return loss

    def backward(self):
        """
        dL/dlogits: (B, C)
        """
        B = self.probs.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(B), self.y] -= 1.0   
        dlogits /= B
        return dlogits

# Take 50% from EACH CLASS (stratified)
def stratified_half(x, y, rng=42):
    rs = np.random.RandomState(rng)
    keep_idx = []
    for c in range(10):  
        idx_c = np.where(y == c)[0]
        rs.shuffle(idx_c)
        k = len(idx_c) // 2  
        keep_idx.append(idx_c[:k])
    keep_idx = np.concatenate(keep_idx)
    rs.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]

train = thv.datasets.MNIST("./", download=True, train=True)
val   = thv.datasets.MNIST("./", download=True, train=False)
#print("Raw:", train.data.shape, len(train.targets), "|", val.data.shape, len(val.targets))

# Convert to NumPy instead of torch tensor
x_tr = train.data.numpy()      
y_tr = train.targets.numpy()   
x_va = val.data.numpy()        
y_va = val.targets.numpy()

x_tr_sub, y_tr_sub = stratified_half(x_tr, y_tr, rng=123)
x_va_sub, y_va_sub = stratified_half(x_va, y_va, rng=456)



#check gradient
def linear_forward_scalar_k(h_l, W, B, k):
    """
    h^{l+1}_k = h_l @ W[k,:].T + B[k]
    h_l: (1,a), W: (c,a), B: (c,),  k: int in [0,c)
    """
    return float(h_l @ W[k, :].T + B[k])

def check_linear_grads(layer, in_dim=392, out_dim=10, eps=1e-5,
                       num_w_checks=10, num_b_checks=5, num_x_checks=10,
                       num_k=5, seed=0):
    rs = np.random.RandomState(seed)

    # test image input
    h_l = rs.randn(1, in_dim).astype(np.float64)
    layer.w = layer.w.astype(np.float64)
    layer.b = layer.b.astype(np.float64)

    # random k
    Ks = rs.choice(out_dim, size=min(num_k, out_dim), replace=False)

    for k in Ks:
        print(f"\n=== Checking grads for output index k={k} ===")

        # running forward/backward get gradient of h^{l+1}_k
        _ = layer.forward(h_l)                   
        dh_next = np.zeros((1, out_dim), dtype=np.float64)
        dh_next[0, k] = 1.0                      # e_k
        dh_l = layer.backward(dh_next)           

        # --------- check dW ----------
        print("Check dW (finite diff vs. backward):")
        for _ in range(num_w_checks):
            i = rs.randint(0, out_dim)          
            j = rs.randint(0, in_dim)            
            old = layer.w[i, j]

            layer.w[i, j] = old + eps
            f_pos = linear_forward_scalar_k(h_l, layer.w, layer.b, k)
            layer.w[i, j] = old - eps
            f_neg = linear_forward_scalar_k(h_l, layer.w, layer.b, k)
            layer.w[i, j] = old

            num = (f_pos - f_neg) / (2 * eps)    
            ana = layer.dw[i, j]                 
            rel_err = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            print(f"  W[{i},{j}]  num={num:+.6e}  ana={ana:+.6e}  rel_err={rel_err:.3e}")

        # --------- check db ----------
        print("Check db (finite diff vs. backward):")
        for _ in range(num_b_checks):
            i = rs.randint(0, out_dim)
            old = layer.b[i]

            layer.b[i] = old + eps
            f_pos = linear_forward_scalar_k(h_l, layer.w, layer.b, k)
            layer.b[i] = old - eps
            f_neg = linear_forward_scalar_k(h_l, layer.w, layer.b, k)
            layer.b[i] = old

            num = (f_pos - f_neg) / (2 * eps)
            ana = layer.db[i]
            rel_err = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            print(f"  b[{i}]     num={num:+.6e}  ana={ana:+.6e}  rel_err={rel_err:.3e}")

        # --------- check d h^l ----------
        print("Check d h^l (finite diff vs. backward):")
        for _ in range(num_x_checks):
            j = rs.randint(0, in_dim)
            old = h_l[0, j]

            h_l[0, j] = old + eps
            f_pos = linear_forward_scalar_k(h_l, layer.w, layer.b, k)
            h_l[0, j] = old - eps
            f_neg = linear_forward_scalar_k(h_l, layer.w, layer.b, k)
            h_l[0, j] = old

            num = (f_pos - f_neg) / (2 * eps)
            ana = dh_l[0, j]
            rel_err = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            print(f"  h_l[0,{j}] num={num:+.6e}  ana={ana:+.6e}  rel_err={rel_err:.3e}")

fc = linear_t(in_dim=392, out_dim=10, seed=1)
check_linear_grads(fc,
                   in_dim=392, out_dim=10,
                   eps=1e-5,
                   num_w_checks=10,  
                   num_b_checks=5,   
                   num_x_checks=10,  
                   num_k=5,          
                   seed=0)
