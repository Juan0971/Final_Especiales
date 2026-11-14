import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class AudioSignal :
    def __init__(self, file_path=None):
        self.samples = None
        self.sample_rate = None
        if file_path:
            self.load_from_file(file_path)
    def load_from_file (self, path):
        self.samples, self.sample_rate = sf.read(path)
        if len(self.samples.shape) > 1:
         self.samples = np.mean(self.samples, axis=1)
    def save_to_file(self, path):
       if self.samples is not None:
         sf.write(path, self.samples, self.sample_rate)
    def get_time_vector(self):
       if self.samples is not None:
          return np.linspace(0, len(self.samples)/ self.sample_rate, num=len(self.samples))
       return None
