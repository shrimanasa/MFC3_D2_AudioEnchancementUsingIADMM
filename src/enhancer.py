import numpy as np
from src.nmf_solver import iadmm_nmf

def unsupervised_separation(V, n_components=40, n_iter=2000):
    """
    Perform blind source separation since ground-truth clean dictionaries are unavailable.
    """
    print("Extracting unsupervised components...")
    W, H, cost = iadmm_nmf(V, n_components=n_components, n_iter=n_iter)
    
    # Cluster bases: Speech usually has bursty, sparse activations. Noise represents continuous bands.
    # We sort by the variance of activation (H matrix rows)
    variances = np.var(H, axis=1)
    
    # Select the top 50% highest variance bases as Speech, rest as Noise
    num_speech = int(0.5 * n_components)
    sorted_idx = np.argsort(variances)[::-1]
    
    speech_idx = sorted_idx[:num_speech]
    noise_idx = sorted_idx[num_speech:]
    
    W_speech = W[:, speech_idx]
    H_speech = H[speech_idx, :]
    V_speech = np.dot(W_speech, H_speech)
    
    W_noise = W[:, noise_idx]
    H_noise = H[noise_idx, :]
    V_noise = np.dot(W_noise, H_noise)
    
    # Wiener Filter mask for better perceptual extraction (reduces artifacts)
    eps = 1e-10
    mask_speech = V_speech / (V_speech + V_noise + eps)
    V_enhanced = V * mask_speech
    
    return V_enhanced, V_noise
