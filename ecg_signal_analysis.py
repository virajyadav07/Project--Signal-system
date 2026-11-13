import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

# ==========================================
# ECG SIGNAL FILTERING AND ANALYSIS PROJECT
# ==========================================

# 1. Generate a sample ECG-like signal
fs = 250                 # Sampling frequency in Hz
duration = 10             # Duration in seconds
t = np.arange(0, duration, 1 / fs)

# Create an ECG-like waveform using Gaussian pulses
ecg_clean = np.zeros_like(t)
heartbeat_interval = 1.0  # Approximately 60 beats per minute

for beat in np.arange(0.5, duration, heartbeat_interval):
    # P wave
    ecg_clean += 0.12 * np.exp(
        -((t - (beat - 0.20)) / 0.045) ** 2
    )

    # Q wave
    ecg_clean -= 0.15 * np.exp(
        -((t - (beat - 0.045)) / 0.015) ** 2
    )

    # R wave
    ecg_clean += 1.00 * np.exp(
        -((t - beat) / 0.018) ** 2
    )

    # S wave
    ecg_clean -= 0.25 * np.exp(
        -((t - (beat + 0.045)) / 0.020) ** 2
    )

    # T wave
    ecg_clean += 0.30 * np.exp(
        -((t - (beat + 0.25)) / 0.080) ** 2
    )

# 2. Add simulated noise
rng = np.random.default_rng(42)

baseline_wander = 0.20 * np.sin(2 * np.pi * 0.3 * t)
high_frequency_noise = 0.08 * rng.normal(size=len(t))
powerline_noise = 0.04 * np.sin(2 * np.pi * 50 * t)

ecg_noisy = (
    ecg_clean
    + baseline_wander
    + high_frequency_noise
    + powerline_noise
)

# 3. Design a Butterworth band-pass filter
low_cutoff = 0.5
high_cutoff = 40

sos = butter(
    4,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs,
    output="sos"
)

# 4. Apply the filter
ecg_filtered = sosfiltfilt(sos, ecg_noisy)

# 5. Calculate SNR (possible because the clean
# reference signal is known in this simulation)
def calculate_snr(reference, test_signal):
    noise = reference - test_signal
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)

snr_before = calculate_snr(ecg_clean, ecg_noisy)
snr_after = calculate_snr(ecg_clean, ecg_filtered)

print("ECG Signal Analysis Results")
print("-" * 35)
print(f"Sampling frequency: {fs} Hz")
print(f"Signal duration: {duration} seconds")
print(f"SNR before filtering: {snr_before:.2f} dB")
print(f"SNR after filtering:  {snr_after:.2f} dB")

# 6. Detect approximate R peaks
peaks, _ = find_peaks(
    ecg_filtered,
    distance=int(0.4 * fs),
    prominence=0.4
)

print(f"Detected R peaks: {len(peaks)}")

# 7. Plot original, noisy and filtered signals
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(t, ecg_clean)
plt.title("Original Synthetic ECG Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, ecg_noisy)
plt.title("ECG Signal with Simulated Noise")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, ecg_filtered, label="Filtered ECG")
plt.plot(
    t[peaks],
    ecg_filtered[peaks],
    "x",
    label="Detected R peaks"
)
plt.title("Filtered ECG Signal and R-Peak Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. Compare frequency spectra
freq = np.fft.rfftfreq(len(t), d=1 / fs)

noisy_spectrum = np.abs(np.fft.rfft(ecg_noisy))
filtered_spectrum = np.abs(np.fft.rfft(ecg_filtered))

plt.figure(figsize=(12, 5))
plt.plot(freq, noisy_spectrum, label="Noisy ECG")
plt.plot(freq, filtered_spectrum, label="Filtered ECG")
plt.title("Frequency Spectrum Before and After Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 80)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


