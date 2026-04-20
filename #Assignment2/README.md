# Keyword Spotting on Arduino Nano 33 BLE Sense Lite

On-device keyword spotting using MFCC feature extraction (CMSIS-DSP) and a quantized CNN (TFLite Micro). The system listens through the onboard PDM microphone and classifies short sounds in real time: **clap**, **tap**, **snap**, and **silence**.

## Hardware

- Arduino Nano 33 BLE Sense Lite (nRF52840, Cortex-M4F, 64 MHz)
- Onboard PDM microphone
- Micro-USB cable

## Software Requirements

- [Arduino Cloud](https://cloud.arduino.cc/) account
- Google Colab (for training)
- Python dependencies: `tensorflow`, `numpy`, `scipy`, `librosa`, `soundfile`, `matplotlib`, `scikit-learn`, `seaborn`, `pydub`
- Arduino library: **Harvard_TinyMLx** (includes TFLite Micro for Nano 33 BLE)

## MFCC Pipeline

The entire MFCC pipeline runs on-device using CMSIS-DSP functions. The same pipeline is replicated in the training notebook using `scipy.fft` and manual filterbank construction — no `librosa.feature.mfcc` is used, to avoid numerical mismatches.

The processing steps are:

1. **PCM conversion** — int16 samples from PDM mic are normalized to float32 in [-1, 1]
2. **Pre-emphasis** — per-frame high-pass filter: `y[i] = x[i] - 0.97 * x[i-1]`
3. **Hamming window** — `w[i] = 0.54 - 0.46 * cos(2π·i / (N-1))`, N=256
4. **RFFT** — 256-point real FFT via `arm_rfft_fast_f32`, producing 129 complex bins
5. **Power spectrum** — `|X[k]|²` using `arm_cmplx_mag_squared_f32`
6. **Mel filterbank** — 26 triangular filters spanning 300–8000 Hz, no Slaney normalization
7. **Log energy** — natural logarithm: `ln(mel + 1e-9)`
8. **DCT-II** — 13 cepstral coefficients, no orthonormal normalization
9. **Per-coefficient normalization** — `(mfcc - mean) / std` using statistics from training
10. **Transposition** — stored as [13 × 124] (coefficient-major) to match CNN input shape

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 kHz |
| Frame length | 256 samples (16 ms) |
| Hop length | 128 samples (8 ms, 50% overlap) |
| Mel filters | 26 |
| MFCC coefficients | 13 |
| Mel range | 300 – 8000 Hz |
| Frames per 1s clip | 124 |

## Model Architecture

A lightweight CNN trained in TensorFlow/Keras and quantized to INT8 via TFLite:

```
Input (1, 13, 124, 1)
  → Conv2D(16, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.3)
  → Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.3)
  → GlobalAveragePooling2D
  → Dense(64, ReLU) + Dropout(0.4)
  → Dense(4, Softmax)
```

The model is fully quantized (INT8 weights and activations) and runs inference in about 34 KB of tensor arena on the Nano 33 BLE.

## How to Run

### Step 1 — Collect audio samples

Record short 1-second WAV files (mono, 16 kHz) for each class. We used 10 original recordings per class. Organize them like this:

```
dataset_fixed/
├── clap/     → clap1.wav, clap2.wav, ...
├── tap/      → tap1.wav, tap2.wav, ...
├── snap/     → snap1.wav, snap2.wav, ...
└── silence/  → silence1.wav, silence2.wav, ...
```

Zip the folder and upload it to Google Colab.

### Step 2 — Train the model in Google Colab

1. Open `MFCC_Arduino_Training_v3.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload the dataset ZIP when prompted.
3. Run all cells. The notebook will:
   - Apply data augmentation (noise, time stretch, pitch shift, volume) to reach 250 samples per class.
   - Extract MFCC features using the CMSIS-identical pipeline (scipy + manual filterbank).
   - Train the CNN with early stopping and learning rate scheduling.
   - Quantize the model to INT8 and verify accuracy.
   - Generate `model.h` containing the TFLite model, normalization constants, and class labels.
4. Download `model.h`.

> A standalone Python script `mfcc_arduino_training_v3.py` is also provided as an alternative.

### Step 3 — Upload to Arduino

1. Open [Arduino Cloud](https://cloud.arduino.cc/) Editor.
2. Create a new sketch or import `KEYWORD_SPOTTING_PEDRINI_BELLINI/keyword_spotting_Pedrini_Bellini.ino`.
3. Add `model.h` as a new tab in the sketch.
4. Add the **Harvard_TinyMLx** library from the Libraries panel.
5. Select board: **Arduino Nano 33 BLE**.
6. Compile and upload.
7. Open Serial Monitor at 115200 baud.

### Step 4 — Test

The board continuously listens and runs inference every second with a 0.5s sliding window overlap. Make a clap, tap, or snap near the microphone and check the Serial Monitor output:

```
[KWS] clap: 0.527  |  tap: 0.160  |  snap: 0.289  |  silence: 0.023
  --> Guess: clap  (confidence = 52.7 %)
```

## Results

Testing was done in a university study room with background noise (people talking). The model correctly distinguishes between the four classes when the sound is made close to the microphone. Some representative outputs:

| Action | Top prediction | Confidence |
|--------|---------------|------------|
| Clap | clap | 53–56% |
| Tap (on desk) | tap | 57–70% |
| Snap | snap | 50–57% |
| Ambient noise | snap/silence | 25–30% |

Silence detection is weaker because the training samples were recorded in a quiet environment, while the test environment had constant background noise. Adding silence samples from noisy environments would improve this.

## Project Structure

```
├── MFCC_Arduino_Training_v3.ipynb          # Colab notebook (training + export)
├── mfcc_arduino_training_v3.py             # Standalone Python script (alternative)
├── dataset_fixed/
│   ├── clap/       (10 original WAV files)
│   ├── tap/        (10 original WAV files)
│   ├── snap/       (10 original WAV files)
│   └── silence/    (10 original WAV files)
└── KEYWORD_SPOTTING_PEDRINI_BELLINI/
    ├── keyword_spotting_Pedrini_Bellini.ino # Arduino sketch
    └── model.h                              # Generated model + parameters
```

## Key Design Decisions

- **No librosa for MFCC extraction in training.** We initially used `librosa.feature.mfcc` but found that its internal defaults (Slaney normalization, orthonormal DCT, center padding) produce values that don't match the CMSIS-DSP implementation on Arduino. We rewrote the extraction pipeline using `scipy.fft.rfft` and manual filterbank/DCT construction to get an exact match.

- **Per-coefficient normalization.** MFCC coefficient 0 (energy) has a much larger magnitude than higher coefficients. A global scalar normalization would distort the features, so we normalize each coefficient independently using mean and standard deviation computed over the training set.

- **INT8 quantization.** The model is fully quantized to INT8 for both weights and activations, reducing memory usage and enabling efficient inference on the Cortex-M4F without floating-point overhead in the neural network layers.

---

**Andrea Pedrini** · **Pietro Bellini**
