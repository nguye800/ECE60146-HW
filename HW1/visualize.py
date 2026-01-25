import numpy as np
import matplotlib.pyplot as plt

from SineWaveFunction import SineWaveFunction
from SquareWaveFunction import SquareWaveFunction
from CompositeSignalFunction import CompositeSignalFunction


def generate_signals(duration):
    sine = SineWaveFunction(amplitude=1.0, frequency=0.1)
    square = SquareWaveFunction(amplitude=0.5, frequency=0.05)
    composite = CompositeSignalFunction(inputs=[sine, square])
    composite(duration)

    return {
        "Sine": np.array(sine.data, dtype=float),
        "Square": np.array(square.data, dtype=float),
        "Composite": np.array(composite.data, dtype=float),
    }


def compute_fft(signal, sample_spacing):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=sample_spacing)
    fft_vals = np.fft.fft(signal)
    mask = freqs >= 0
    return freqs[mask], np.abs(fft_vals)[mask] / n


def plot_time_domain(ax, time_axis, signals):
    for label, data in signals.items():
        if label == "Sine":
            ax.plot(time_axis, data, label=label, color="tab:blue", linestyle="-")
        elif label == "Square":
            ax.plot(time_axis, data, label=label, color="tab:orange", linestyle="--")
        elif label == "Composite":
            ax.plot(time_axis, data, label=label, color="tab:green", linestyle="-.")
        else:
            ax.plot(time_axis, data, label=label)

    ax.set_title("Signal Comparison (Time Domain)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

def plot_frequency_domain(ax, signals, sample_spacing):
    for label, data in signals.items():
        frequencies, magnitude = compute_fft(data, sample_spacing)
        ax.plot(frequencies, magnitude, label=label)
    ax.set_xlim(left=0)
    ax.set_title("Signal Comparison (Frequency Domain)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)


def main():
    duration = 50
    sample_spacing = 1.0
    time_axis = np.arange(duration) * sample_spacing
    signals = generate_signals(duration)

    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    plot_time_domain(ax_time, time_axis, signals)
    plot_frequency_domain(ax_freq, signals, sample_spacing)
    plt.show()


if __name__ == "__main__":
    main()
