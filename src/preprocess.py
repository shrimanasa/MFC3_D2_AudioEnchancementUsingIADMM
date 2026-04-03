import librosa
import numpy as np
import av

def load_audio(file_path, sr=8000):
    container = av.open(file_path)
    stream = container.streams.audio[0]
    
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        frames.append(arr)
        
    audio = np.concatenate(frames, axis=1)
    
    # If stereo, average to mono
    if audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)
    else:
        audio = audio[0]
        
    orig_sr = stream.rate
    if orig_sr != sr:
        audio = librosa.resample(audio.astype(float), orig_sr=orig_sr, target_sr=sr)
        
    return audio

def get_stft(audio, n_fft=1024, hop_length=256):
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.exp(1.j * np.angle(D))
    return magnitude, phase

def get_istft(magnitude, phase, hop_length=256):
    D = magnitude * phase
    audio = librosa.istft(D, hop_length=hop_length)
    return audio
