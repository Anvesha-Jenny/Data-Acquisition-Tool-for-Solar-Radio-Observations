import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import threading
import time
from astropy.io import fits
import socket


class SpectrogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Spectrogram")
        self.root.geometry("1000x600")

        # Input fields
        tk.Label(root, text="Sampling Frequency (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.fs_entry = tk.Entry(root)
        self.fs_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(root, text="FFT Points (N):").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.n_entry = tk.Entry(root)
        self.n_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(root, text="TCP Port:").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.port_entry = tk.Entry(root)
        self.port_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.port_entry.insert(0, "9999")

        self.start_button = tk.Button(root, text="Start", command=self.start_observation)
        self.start_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.pause_button = tk.Button(root, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        self.save_button = tk.Button(root, text="Save", command=self.save_outputs, state=tk.DISABLED)
        self.save_button.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

        self.close_button = tk.Button(root, text="Close", command=self.close_app)
        self.close_button.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        root.grid_rowconfigure(4, weight=1)
        for i in range(3):
            root.grid_columnconfigure(i, weight=1)

        self.colorbar = None
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.data = []
        self.timestamps = []

    def start_observation(self):
        try:
            self.fs = int(self.fs_entry.get())
            self.n = int(self.n_entry.get())
            self.port = int(self.port_entry.get())
            assert self.fs > 0 and self.n > 0
        except:
            messagebox.showerror("Invalid Input", "Please enter valid integers for Fs, N, and Port.")
            return

        self.freqs = np.linspace(0, self.fs / 2, self.n)
        self.running = True
        self.paused = False
        self.data = []
        self.timestamps = []

        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

        threading.Thread(target=self.receive_data_tcp, daemon=True).start()
        threading.Thread(target=self.auto_save, daemon=True).start()
        self.update_plot()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def receive_data_tcp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', self.port))
        sock.listen(1)
        print(f"Waiting for TCP connection on port {self.port}...")
        conn, addr = sock.accept()
        print(f"Connected to {addr}")

        buffer = b''
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                buffer += conn.recv(4096)
                while len(buffer) >= self.n * 4:
                    sample = np.frombuffer(buffer[:self.n * 4], dtype=np.float32)
                    buffer = buffer[self.n * 4:]

                    with self.lock:
                        self.data.append(sample)
                        self.timestamps.append(time.time())

                        # Keep only the last 10 seconds
                        cutoff = time.time() - 10
                        while self.timestamps and self.timestamps[0] < cutoff:
                            self.timestamps.pop(0)
                            self.data.pop(0)
            except Exception as e:
                print("TCP read error:", e)
                break

        conn.close()
        sock.close()

    def update_plot(self):
        if not self.running:
            return
        if self.paused:
            self.root.after(100, self.update_plot)
            return

        current_time = time.time()

        with self.lock:
            if len(self.data) == 0:
                self.root.after(100, self.update_plot)
                return

            times = np.array(self.timestamps)
            rel_times = times - current_time  # -10 to 0
            data_array = np.array(self.data)

        duration = 10
        num_steps = int(duration / 0.05)
        full_times = np.linspace(-10, 0, num_steps)
        aligned_data = np.full((num_steps, self.n), np.nan)

        for t, d in zip(rel_times, data_array):
            idx = np.searchsorted(full_times, t)
            if 0 <= idx < num_steps:
                aligned_data[idx] = d

        extent = [-10, 0, 0, self.fs / 2]
        self.ax.clear()
        im = self.ax.imshow(
            aligned_data.T,
            aspect='auto',
            extent=extent,
            origin='lower',
            cmap='nipy_spectral',  # Rainbow-like color map
            interpolation='nearest'
        )
        self.ax.set_title(f"Live Spectrogram (Last 10s) - Fs: {self.fs} Hz, N: {self.n}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_xlim(-10, 0)
        self.ax.set_ylim(0, self.fs / 2)

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label="Intensity")
        else:
            im.set_clim(vmin=np.nanmin(aligned_data), vmax=np.nanmax(aligned_data))
            self.colorbar.update_normal(im)

        self.canvas.draw()
        self.root.after(100, self.update_plot)

    def auto_save(self):
        while self.running:
            time.sleep(60)
            if self.running and not self.paused:
                self.save_outputs()

    def save_outputs(self):
        with self.lock:
            if not self.data:
                return
            rel_times = np.array(self.timestamps) - self.timestamps[-1]
            data_array = np.array(self.data)

        base_name = datetime.now().strftime("Udaipur_PRL_%d%m%Y_%H%M%S") + f"_Fs{self.fs}_N{self.n}"
        fits_name = base_name + ".fits"
        png_name = base_name + ".png"

        hdu_time = fits.PrimaryHDU(rel_times)
        hdu_time.header['FS'] = self.fs
        hdu_time.header['NFFT'] = self.n
        hdu_freq = fits.ImageHDU(self.freqs, name="FREQ")
        hdu_data = fits.ImageHDU(data_array, name="DATA")
        fits.HDUList([hdu_time, hdu_freq, hdu_data]).writeto(fits_name, overwrite=True)
        print(f"Saved FITS: {fits_name}")

        fig, ax = plt.subplots(figsize=(10, 4))
        extent = [rel_times[0], rel_times[-1], 0, self.fs / 2]
        im = ax.imshow(data_array.T, aspect='auto', extent=extent, origin='lower', cmap='nipy_spectral')
        ax.set_title(f"60s Spectrogram Snapshot - Fs: {self.fs} Hz, N: {self.n}")
        ax.set_xlabel("Time (s)\n" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout()
        plt.savefig(png_name)
        plt.close()
        print(f"Saved PNG: {png_name}")

    def close_app(self):
        if self.running:
            self.running = False
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.mainloop()
