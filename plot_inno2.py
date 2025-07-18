import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime
import threading
import time
from astropy.io import fits


class SpectrogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Spectrogram")
        self.root.geometry("1200x700")

        # Input fields
        tk.Label(root, text="Sampling Frequency (Hz):").grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.fs_entry = tk.Entry(root)
        self.fs_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        tk.Label(root, text="FFT Points (N):").grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.n_entry = tk.Entry(root)
        self.n_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.start_button = tk.Button(root, text="Start Observation", command=self.start_observation)
        self.start_button.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.stop_button = tk.Button(root, text="Stop Observation", command=self.stop_observation, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        self.close_button = tk.Button(root, text="Close", command=self.close_app)
        self.close_button.grid(row=2, column=2, sticky="ew", padx=5, pady=5)

        tk.Label(root, text="Color Map:").grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        self.cmap_var = tk.StringVar(value='viridis')
        self.cmap_combo = ttk.Combobox(root, textvariable=self.cmap_var, values=plt.colormaps())
        self.cmap_combo.grid(row=0, column=3, padx=5, pady=5)
        self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        # Plot and toolbar frame
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        root.grid_rowconfigure(3, weight=1)
        for i in range(4):
            root.grid_columnconfigure(i, weight=1)

        self.colorbar = None
        self.running = False
        self.lock = threading.Lock()

        self.live_duration = 10
        self.save_duration = 60
        self.avg_duration = 30
        self.data_interval = 0.02
        self.update_interval = 100

        self.avg_label = tk.Label(root, text="Avg Frequency (30s): --- Hz")
        self.avg_label.grid(row=1, column=2, columnspan=2, sticky="ew", padx=5, pady=5)

    def start_observation(self):
        try:
            self.fs = int(self.fs_entry.get())
            self.n = int(self.n_entry.get())
            assert self.fs > 0 and self.n > 0
        except Exception:
            messagebox.showerror("Invalid Input", "Please enter positive integers for Fs and N.")
            return

        self.freqs = np.linspace(0, self.fs / 2, self.n)
        self.running = True

        self.live_steps = int(self.live_duration / self.data_interval)
        self.save_steps = int(self.save_duration / self.data_interval)
        self.avg_steps = int(self.avg_duration / self.data_interval)

        self.live_buffer = np.zeros((self.live_steps, self.n))
        self.save_buffer = np.zeros((self.save_steps, self.n))
        self.avg_buffer = np.zeros((self.avg_steps, self.n))

        self.live_index = self.save_index = self.avg_index = 0

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        threading.Thread(target=self.generate_data, daemon=True).start()
        threading.Thread(target=self.auto_save, daemon=True).start()
        self.update_plot()

    def generate_data(self):
        while self.running:
            new_sample = np.random.rand(self.n) * 100

            with self.lock:
                self.live_buffer[self.live_index, :] = new_sample
                self.live_index = (self.live_index + 1) % self.live_steps

                self.save_buffer[self.save_index, :] = new_sample
                self.save_index = (self.save_index + 1) % self.save_steps

                self.avg_buffer[self.avg_index, :] = new_sample
                self.avg_index = (self.avg_index + 1) % self.avg_steps

            time.sleep(self.data_interval)

    def update_plot(self):
        if not self.running:
            return

        with self.lock:
            if np.all(self.live_buffer == 0):
                self.root.after(self.update_interval, self.update_plot)
                return

            if self.live_index == 0:
                data_to_plot = self.live_buffer.copy()
            else:
                data_to_plot = np.vstack((self.live_buffer[self.live_index:, :], self.live_buffer[:self.live_index, :]))

        extent = [-self.live_duration, 0, 0, self.fs / 2]
        self.ax.clear()
        im = self.ax.imshow(
            data_to_plot.T,
            aspect='auto',
            extent=extent,
            origin='lower',
            cmap=self.cmap_var.get(),
            interpolation='nearest'
        )
        self.ax.set_title("Live Spectrogram (Last 10s)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_xlim(-self.live_duration, 0)
        self.ax.set_ylim(0, self.fs / 2)

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label="Intensity")
        else:
            im.set_clim(vmin=np.min(data_to_plot), vmax=np.max(data_to_plot))
            self.colorbar.update_normal(im)

        self.canvas.draw()

        with self.lock:
            if self.avg_index == 0:
                avg_data = self.avg_buffer.copy()
            else:
                avg_data = np.vstack((self.avg_buffer[self.avg_index:, :], self.avg_buffer[:self.avg_index, :]))
            avg_spectrum = np.mean(avg_data, axis=0)
            peak_freq = self.freqs[np.argmax(avg_spectrum)]
            self.avg_label.config(text=f"Avg Frequency (30s): {peak_freq:.2f} Hz")

        self.root.after(self.update_interval, self.update_plot)

    def auto_save(self):
        while self.running:
            time.sleep(60)
            if self.running:
                self.save_outputs()

    def save_outputs(self):
        with self.lock:
            if np.all(self.save_buffer == 0):
                return

            if self.save_index == 0:
                data_60s = self.save_buffer.copy()
            else:
                data_60s = np.vstack((self.save_buffer[self.save_index:, :], self.save_buffer[:self.save_index, :]))

        rel_times_60s = np.linspace(-self.save_duration, 0, self.save_steps)
        base_name = datetime.now().strftime("Udaipur_PRL_%d%m%Y_%H%M%S") + f"_Fs{self.fs}_N{self.n}"
        fits_name = base_name + ".fits"
        png_name = base_name + ".png"

        hdu_time = fits.PrimaryHDU(rel_times_60s)
        hdu_time.header['FS'] = self.fs
        hdu_time.header['NFFT'] = self.n
        hdu_freq = fits.ImageHDU(self.freqs, name="FREQ")
        hdu_data = fits.ImageHDU(data_60s, name="DATA")
        fits.HDUList([hdu_time, hdu_freq, hdu_data]).writeto(fits_name, overwrite=True)
        print(f"Saved FITS: {fits_name}")

        fig, ax = plt.subplots(figsize=(12, 5))
        extent = [rel_times_60s[0], rel_times_60s[-1], 0, self.fs / 2]
        im = ax.imshow(data_60s.T, aspect='auto', extent=extent, origin='lower', cmap=self.cmap_var.get())
        ax.set_title(f"60s Spectrogram Snapshot  (Fs={self.fs} Hz, N={self.n})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, -0.15, f"Captured at: {now_str}", ha='center', va='top', transform=ax.transAxes, fontsize=10)
        fig.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout()
        plt.savefig(png_name, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved PNG: {png_name}")

    def stop_observation(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_outputs()

    def close_app(self):
        if self.running:
            self.stop_observation()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.mainloop()
