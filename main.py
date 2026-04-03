import os
import glob
import soundfile as sf
import numpy as np

from src.preprocess import load_audio, get_stft, get_istft
from src.enhancer import unsupervised_separation
from src.evaluator import evaluate_proxy_metrics

def main():
    dataset_path = r"C:\Users\shrim\Downloads\archive"
    files = glob.glob(os.path.join(dataset_path, "sample-*.webm"))
    
    if not files:
        print(f"Error: No .webm files found in {dataset_path}")
        return

    output_dir = "enhanced_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(files)} files. Starting Unsupervised Enhancement Pipeline...\n")
    
    for idx, fpath in enumerate(files):
        print(f"--- Processing {os.path.basename(fpath)} ---")
        
        # 1. Load & Preprocess
        audio = load_audio(fpath, sr=8000)
        mag, phase = get_stft(audio)
        
        # 2. Extract Speech via Unsupervised IADMM-NMF and PEWI tracking
        # Using 500 n_iter for demo speed, usually 2000 for strict convergence
        mag_enhanced, mag_noise = unsupervised_separation(mag, n_components=30, n_iter=500) 
        
        # 3. Reconstruct
        enhanced_audio = get_istft(mag_enhanced, phase)
        noise_audio = get_istft(mag_noise, phase)
        
        # Pad differences if istft length varies
        min_len = min(len(audio), len(enhanced_audio))
        audio = audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]
        noise_audio = noise_audio[:min_len]
        
        # 4. Proxy Evaluate
        metrics = evaluate_proxy_metrics(audio, enhanced_audio, noise_audio)
        print("Proxy Metrics:", metrics)
        
        # Check artifact levels
        if metrics.get('SAR', 0) < 5.0 and 'error' not in metrics:
            print("WARNING: High artifacts detected (SAR low). Optimization needed internally.")
            
        # 5. Save Output
        # Increase Volume (Normalize to near-peak to recover any volume strictly lost in filtering)
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0:
            enhanced_audio = (enhanced_audio / max_val) * 0.95
            
        out_name = os.path.join(output_dir, f"enhanced_{os.path.basename(fpath)}.wav")
        sf.write(out_name, enhanced_audio, 8000)
        print(f"Saved: {out_name}\n")

if __name__ == "__main__":
    main()
