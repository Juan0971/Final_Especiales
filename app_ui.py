import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from audio_signal import AudioSignal
from fourier_processor import FourierProcessor
from filters import SpectralSubstractionFilter
class NoiseReducerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reducción de Ruido por Análisis de Fourier")
        self.root.geometry("1100x800")
        self.original_audio = None
        self.cleaned_audio = None
        self.processor = FourierProcessor(nperseg=2048, noverlap=1024)
        self.create_widgets()
    def create_widgets(self):
        control_frame = tk.LabelFrame(self.root, text=" Panel de Control y Procesamiento ", font=("Arial", 10, "bold"))
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=15, pady=15)
        btn_load = tk.Button(control_frame, text="1. Cargar Audio WAV", command=self.load_audio, bg="
        btn_load.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(control_frame, text="Intensidad (Alpha):", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.slider_alpha = tk.Scale(control_frame, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.slider_alpha.set(2.0)
        self.slider_alpha.pack(side=tk.LEFT, padx=5)
        btn_process = tk.Button(control_frame, text="2. Aplicar Filtro Fourier", command=self.process_audio, bg="
        btn_process.pack(side=tk.LEFT, padx=10, pady=10)
        btn_save = tk.Button(control_frame, text="3. Guardar Audio Limpio", command=self.save_audio, bg="
        btn_save.pack(side=tk.LEFT, padx=10, pady=10)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.setup_empty_plots()
    def setup_empty_plots(self):
        """Configura los ejes iniciales en blanco antes de cargar datos."""
        for row in self.axes:
            for ax in row:
                ax.clear()
                ax.set_title("Esperando datos...", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()
    def load_audio(self):
        """Abre el diálogo para seleccionar el archivo de audio con ruido."""
        file_path = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
        if file_path:
            try:
                self.original_audio = AudioSignal(file_path)
                messagebox.showinfo("Éxito", "¡Archivo de audio cargado con éxito!")
                self.axes[0, 0].clear()
                self.axes[0, 0].plot(self.original_audio.get_time_vector(), self.original_audio.samples, color='crimson', alpha=0.7)
                self.axes[0, 0].set_title("Señal Original con Ruido (Tiempo)", fontsize=9)
                self.axes[0, 0].set_xlabel("Tiempo [s]", fontsize=8)
                self.axes[0, 0].set_ylabel("Amplitud", fontsize=8)
                f, t, Zxx = self.processor.compute_stft(self.original_audio)
                self.axes[0, 1].clear()
                self.axes[0, 1].pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.abs(Zxx).max()/10, shading='gouraud')
                self.axes[0, 1].set_title("Espectrograma Original (Frecuencia)", fontsize=9)
                self.axes[0, 1].set_xlabel("Tiempo [s]", fontsize=8)
                self.axes[0, 1].set_ylabel("Frecuencia [Hz]", fontsize=8)
                self.fig.tight_layout()
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el audio: {e}")
    def process_audio(self):
        """Ejecuta la STFT, aplica el filtro espectral y reconstruye con IFFT."""
        if not self.original_audio:
            messagebox.showwarning("Atención", "Por favor carga un archivo de audio primero.")
            return
        try:
            f, t, Zxx_orig = self.processor.compute_stft(self.original_audio)
            alpha_val = self.slider_alpha.get()
            spectral_filter = SpectralSubstractionFilter(noise_duration_sec=0.3, alpha=alpha_val)
            Zxx_clean = spectral_filter.apply(Zxx_orig, f, t)
            self.cleaned_audio = self.processor.compute_istft(Zxx_clean, self.original_audio.sample_rate)
            self.axes[1, 0].clear()
            self.axes[1, 0].plot(self.cleaned_audio.get_time_vector(), self.cleaned_audio.samples, color='dodgerblue', alpha=0.7)
            self.axes[1, 0].set_title("Señal Filtrada / Limpia (Tiempo)", fontsize=9)
            self.axes[1, 0].set_xlabel("Tiempo [s]", fontsize=8)
            self.axes[1, 0].set_ylabel("Amplitud", fontsize=8)
            self.axes[1, 1].clear()
            self.axes[1, 1].pcolormesh(t, f, np.abs(Zxx_clean), vmin=0, vmax=np.abs(Zxx_orig).max()/10, shading='gouraud')
            self.axes[1, 1].set_title("Espectrograma Limpio (Frecuencia)", fontsize=9)
            self.axes[1, 1].set_xlabel("Tiempo [s]", fontsize=8)
            self.axes[1, 1].set_ylabel("Frecuencia [Hz]", fontsize=8)
            self.fig.tight_layout()
            self.canvas.draw()
            messagebox.showinfo("Completado", "¡Procesamiento por Fourier finalizado con éxito!")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento: {e}")
    def save_audio(self):
        """Guarda el archivo resultante limpio."""
        if not self.cleaned_audio:
            messagebox.showwarning("Atención", "No hay audio procesado disponible para guardar.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Archivos WAV", "*.wav")])
        if file_path:
            self.cleaned_audio.save_to_file(file_path)
            messagebox.showinfo("Guardado", "El archivo limpio se guardó correctamente.")
