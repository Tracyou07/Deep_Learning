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

class SimpleMNISTModel:
    def __init__(self):
        self.embed = embedding_t()
        self.fc1   = linear_t(in_dim=392, out_dim=128) 
        self.relu  = relu_t()
        self.fc2   = linear_t(in_dim=128, out_dim=10)
        self.lossf = softmax_ce_loss()

    def zero_grad(self):
        self.embed.zero_grad()
        self.fc1.zero_grad()
        self.fc2.zero_grad()

    def forward(self, x):             # x:(B,28,28) -> logits:(B,10)
        h  = self.embed.forward(x)    # (B,392)
        z1 = self.fc1.forward(h)      # (B,128)
        a1 = self.relu.forward(z1)    # (B,128)
        logits = self.fc2.forward(a1) # (B,10)
        return logits

    def backward(self, dlogits):
        da1 = self.fc2.backward(dlogits)   # (B,128)
        dz1 = self.relu.backward(da1)      # (B,128)
        dh  = self.fc1.backward(dz1)       # (B,392)
        _   = self.embed.backward(dh)      # (B,28,28)

    def step(self, lr=0.1):
        # SGD 
        self.embed.w -= lr * self.embed.dw; self.embed.b -= lr * self.embed.db
        self.fc1.w   -= lr * self.fc1.dw;   self.fc1.b   -= lr * self.fc1.db
        self.fc2.w   -= lr * self.fc2.dw;   self.fc2.b   -= lr * self.fc2.db

def validate(model, x_val, y_val, batch_size=32):
  
    N = len(y_val)
    total_loss = 0.0
    total_correct = 0

    # sequential
    for i in range(0, N, batch_size):
        xb = x_val[i:i+batch_size]
        yb = y_val[i:i+batch_size]

        logits = model.forward(xb)          
        loss = model.lossf.forward(logits, yb)
        total_loss += loss * len(yb)
        preds = np.argmax(logits, axis=1)
        total_correct += np.sum(preds == yb)

    avg_loss = total_loss / N
    avg_error = 1.0 - (total_correct / N)
    return avg_loss, avg_error



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

# x: (N,28,28) uint8 -> float32 in [0,1]
x_tr_sub = (x_tr_sub.astype(np.float32) / 255.0)
x_va_sub = (x_va_sub.astype(np.float32) / 255.0)
y_tr_sub = y_tr_sub.astype(np.int64)
y_va_sub = y_va_sub.astype(np.int64)

# training loop
def batch_iter(X, Y, bs, rng):
    n = len(Y)
    idx = rng.choice(n, size=bs, replace=False)
    return X[idx], Y[idx]

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)

rng = np.random.RandomState(0)
model = SimpleMNISTModel()
lr = 0.1
batch_size = 64
updates = 50000   # steps

val_steps = []
val_losses = []
val_errors = []

for t in range(1, updates+1):
    xb, yb = batch_iter(x_tr_sub, y_tr_sub, batch_size, rng)

    model.zero_grad()
    logits = model.forward(xb)
    loss = model.lossf.forward(logits, yb)
    dlogits = model.lossf.backward()
    model.backward(dlogits)
    model.step(lr)

    if t % 1000 == 0:
        vloss, verr = validate(model, x_va_sub, y_va_sub, batch_size=32)
        val_steps.append(t)
        val_losses.append(vloss)
        val_errors.append(verr)
        print(f"[val @ {t}] loss={vloss:.4f}, error={verr:.3f}")

plt.figure(figsize=(8,4))
plt.plot(val_steps, val_losses, marker='o', label='val loss')
plt.xlabel('weight updates')
plt.ylabel('loss')
plt.title('Validation loss vs updates (every 1000)')
plt.grid(True); plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(val_steps, val_errors, marker='o', label='val error')
plt.xlabel('weight updates')
plt.ylabel('error rate')
plt.title('Validation error vs updates (every 1000)')
plt.grid(True); plt.legend()
plt.show()