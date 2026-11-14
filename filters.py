from abc import ABC, abstractmethod
import numpy as np

class SpectralSubstractionFilter:
    def __init__(self, noise_duration_sec=0.3, alpha=1.5):
        self.noise_duration_sec = noise_duration_sec
        self.alpha= alpha

    def apply(self, Zxx, frequencies, times):
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        sample_rate_aprox = len(times) / times[-1] if len(times) > 0 and times[-1] > 0 else 1
        noise_frames= int(self.noise_duration_sec * sample_rate_aprox)
        noise_frames = max(1, min(noise_frames, magnitude.shape[1]))

        noise_profile = np.mean(magnitude[:, :noise_frames], axis= 1, keepdims= True)

        substracted_magnitude = magnitude - (self.alpha * noise_profile)
        cleaned_magnitude = np.maximum(substracted_magnitude, 0)

        Zxx_cleaned = cleaned_magnitude * np.exp(1j * phase)
        return Zxx_cleaned 