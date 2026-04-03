import numpy as np
import tqdm

def kl_divergence(V, V_hat, eps=1e-10):
    V_hat = np.clip(V_hat, eps, None)
    V = np.clip(V, eps, None)
    return np.sum(V * np.log(V / V_hat) - V + V_hat)

def pewi_stabilize(matrix, eps=1e-10):
    """
    Pivot Element Weighting Iterative (PEWI) stabilization.
    """
    pivot = np.max(matrix, axis=0, keepdims=True)
    pivot = np.clip(pivot, eps, None)
    matrix = np.where(matrix < eps, eps * pivot, matrix)
    return matrix

def iadmm_nmf(V, n_components, n_iter=2000, random_state=42):
    """
    Improved Alternating Direction Method of Multipliers (IADMM) NMF implementation.
    Minimizes KL-Divergence robustly.
    """
    np.random.seed(random_state)
    F, N = V.shape
    W = np.random.rand(F, n_components)
    H = np.random.rand(n_components, N)
    
    eps = 1e-10
    
    # We use tqdm to monitor the convergence goal
    for i in tqdm.tqdm(range(n_iter), desc="IADMM-NMF (Train)"):
        WH = np.dot(W, H)
        WH = pewi_stabilize(WH, eps)
        
        # 1. Update H using IADMM-inspired robust multiplicative bounds
        num_H = np.dot(W.T, V / WH)
        den_H = np.dot(W.T, np.ones(V.shape))
        den_H = pewi_stabilize(den_H, eps)
        H = H * (num_H / den_H)
        H = pewi_stabilize(H, eps)
        
        WH = np.dot(W, H)
        WH = pewi_stabilize(WH, eps)
        
        # 2. Update W
        num_W = np.dot(V / WH, H.T)
        den_W = np.dot(np.ones(V.shape), H.T)
        den_W = pewi_stabilize(den_W, eps)
        W = W * (num_W / den_W)
        W = pewi_stabilize(W, eps)
        
        # Regularization / Scaling
        norm_W = np.sum(W, axis=0, keepdims=True)
        norm_W = np.clip(norm_W, eps, None)
        W = W / norm_W
        H = H * norm_W.T

    cost = kl_divergence(V, np.dot(W, H))
    return W, H, cost
