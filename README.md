# EXP 1 :  ANALYSIS OF DFT WITH AUDIO SIGNAL

# AIM: 

  To analyze audio signal by removing unwanted frequency. 

# APPARATUS REQUIRED: 
   
   google Colab

# PROGRAM: 
```
# ==============================
# AUDIO DFT ANALYSIS IN COLAB
# ==============================

# Step 1: Install required packages
!pip install -q librosa soundfile

# Step 2: Upload audio file
from google.colab import files
uploaded = files.upload()   # choose your .wav / .mp3 / .flac file

# Handle file upload
filename = next(iter(uploaded.keys()))
print("Uploaded:", filename)

# Step 3: Load audio
import librosa, librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

y, sr = librosa.load(filename, sr=None, mono=True)  # keep original sample rate
duration = len(y) / sr
print(f"Sample rate = {sr} Hz, duration = {duration:.2f} s, samples = {len(y)}")

# Step 4: Play audio
from IPython.display import Audio, display
display(Audio(y, rate=sr))

# Step 5: Full FFT (DFT) analysis
n_fft = 2**14   # choose large power of 2 for smoother spectrum
Y = np.fft.rfft(y, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, 1/sr)
magnitude = np.abs(Y)

plt.figure(figsize=(12,4))
plt.plot(freqs, magnitude)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum (linear scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.semilogy(freqs, magnitude + 1e-12)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT Magnitude Spectrum (log scale)")
plt.grid(True)
plt.show()

# Step 6: Top 10 dominant frequencies
N = 10
idx = np.argsort(magnitude)[-N:][::-1]
print("\nTop 10 Dominant Frequencies:")
for i, k in enumerate(idx):
    print(f"{i+1:2d}. {freqs[k]:8.2f} Hz  (Magnitude = {magnitude[k]:.2e})")

# Step 7: Spectrogram (STFT)
n_fft = 2048
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.ylim(0, sr/2)
plt.show()

```
#Audio Used:
https://drive.google.com/file/d/1C6G3UKHbNOJo1gcgvfXu0hbLRIzW8RdE/view?usp=sharing


# OUTPUT: 
<img width="1005" height="393" alt="download (2)" src="https://github.com/user-attachments/assets/bc44e748-5ebe-4d8b-b447-70a0368aeb90" />
<img width="1012" height="393" alt="download" src="https://github.com/user-attachments/assets/d6fbbe06-4ef1-4a21-b4a4-ec53efed6343" />

Top 10 Dominant Frequencies:
 1.   146.48 Hz  (Magnitude = 6.31e+02)
 2.   442.38 Hz  (Magnitude = 4.57e+02)
 3.   439.45 Hz  (Magnitude = 4.08e+02)
 4.   149.41 Hz  (Magnitude = 3.22e+02)
 5.   143.55 Hz  (Magnitude = 3.18e+02)
 6.   881.84 Hz  (Magnitude = 2.79e+02)
 7.   445.31 Hz  (Magnitude = 1.95e+02)
 8.   878.91 Hz  (Magnitude = 1.94e+02)
 9.   436.52 Hz  (Magnitude = 1.72e+02)
10.   152.34 Hz  (Magnitude = 1.47e+02)
<img width="958" height="470" alt="download (1)" src="https://github.com/user-attachments/assets/bb4047f9-075d-46dc-9ae5-fce13ab9de29" />



# RESULTS
THUS,THE ANALYSIS OF DFT WITH AUDIO SIGNAL IS VERIFIED
