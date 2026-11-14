import numpy as np
from scipy.signal import stft, istft
from audio_signal import AudioSignal

class FourierProcessor:
    def __init__(self, nperseg=2048, noverlap=1024, window='hann'):
        self.nperseg = nperseg      # Tamaño de la ventana de análisis
        self.noverlap = noverlap    # Solapamiento (Overlap-Add)
        self.window = window

    def compute_stft(self, audio_signal: AudioSignal):
        """Calcula la Transformada de Fourier de Tiempo Corto (STFT)."""
        frequencies, times, Zxx = stft(
            audio_signal.samples, 
            fs=audio_signal.sample_rate, 
            window=self.window, 
            nperseg=self.nperseg, 
            noverlap=self.noverlap
        )
        return frequencies, times, Zxx

    def compute_istft(self, Zxx, sample_rate):
        """Reconstruye la señal temporal mediante la IFFT (Overlap-Add)."""
        _, reconstructed_samples = istft(
            Zxx, 
            fs=sample_rate, 
            window=self.window, 
            nperseg=self.nperseg, 
            noverlap=self.noverlap
        )
        out_signal = AudioSignal()
        out_signal.samples = reconstructed_samples
        out_signal.sample_rate = sample_rate
        return out_signal