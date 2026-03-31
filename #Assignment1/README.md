# Gesture Recognition on Arduino Nano 33 BLE Sense Lite

On-device gesture classification using a custom MLP neural network deployed on an Arduino Nano 33 BLE Sense Lite. The system recognizes five gestures in real time using the onboard IMU (LSM9DS1): **Punch**, **Flex**, **Shake**, **UpDown**, and **Circle**.

## Hardware

- Arduino Nano 33 BLE Sense Lite
- Micro-USB cable
- Onboard IMU: LSM9DS1 (accelerometer + gyroscope)

## Software Requirements

- [Arduino Cloud](https://cloud.arduino.cc/) account (for compiling and uploading the sketch)
- Google Colab
- Python dependencies: `tensorflow`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`

## Architecture Overview

The model is a fully-connected MLP with the following topology:

```
Input (36) → Dense(50, ReLU) → Dense(15, ReLU) → Dense(5, Softmax)
```

36 input features are extracted from a sliding window of 20 IMU samples (6 axes):

| Feature group | Count | Description                          |
|---------------|-------|--------------------------------------|
| Mean          | 6     | Per-axis mean                        |
| Std           | 6     | Per-axis standard deviation          |
| RMS           | 6     | Per-axis root mean square            |
| Min           | 6     | Per-axis minimum                     |
| Max           | 6     | Per-axis maximum                     |
| PSD           | 6     | Per-axis power spectral density (DFT)|

## How to Run

### Step 1 — Train the model and generate `model.h`

1. Open `Gesture_Recognition.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload the CSV files from the `training data/` folder into the Colab runtime (via the file browser on the left panel).
3. Run all cells sequentially. The notebook will:
   - Load and segment the training data into windows.
   - Extract time-domain and frequency-domain features.
   - Train the MLP with TensorFlow/Keras.
   - Export the trained weights, biases, normalization parameters, and gesture class names into a file called `model.h`.
4. Download the generated `model.h` from the Colab file browser.

> **Alternative:** If the notebook does not work correctly, use the standalone Python script `gesture_recognition.py` instead. Run it locally with:
> ```bash
> pip install tensorflow pandas numpy matplotlib scikit-learn
> python gesture_recognition.py
> ```
> It performs the same pipeline and produces the same `model.h` output.

### Step 2 — Upload the sketch to Arduino Cloud and flash the board

1. Go to [Arduino Cloud](https://cloud.arduino.cc/) and open the **Editor**.
2. Create a new sketch or import the existing sketch located at:
   ```
   Gesture-Recognition_Pedrini_Bellini/Gesture-Recognition_Pedrini_Bellini.ino
   ```
3. Add the generated `model.h` file as a new tab in the sketch (click the **"..."** menu → **Add Tab** → name it `model.h` and paste the content).
4. Make sure the board selected is **Arduino Nano 33 BLE**.
5. Click **Verify** to compile the sketch.
6. Connect the Arduino Nano 33 BLE Sense Lite via USB and click **Upload**.
7. Open the **Serial Monitor** at 9600 baud to see classification output.

### Step 3 — Perform gestures

Once the sketch is running:

- The board waits for a motion trigger (acceleration magnitude > 2.5 g).
- When motion is detected, it captures a window of 20 IMU samples.
- Features are extracted, normalized, and fed through the MLP.
- The predicted gesture and confidence are printed to the Serial Monitor.
- A cooldown of 500 ms prevents duplicate classifications.

## Project Structure

```
├── Gesture_Recognition.ipynb                          # Colab notebook (training + export)
├── gesture_recognition.py                             # Standalone Python script (alternative)
├── training data/
│   ├── Circle.csv
│   ├── Flex.csv
│   ├── Punch.csv
│   ├── Shake.csv
│   └── UpDown.csv
└── Gesture-Recognition_Pedrini_Bellini/
    ├── Gesture-Recognition_Pedrini_Bellini.ino        # Arduino sketch
    ├── model.h                                        # Auto-generated model parameters
    └── sketch.json                                    # Board configuration
```

---

**Andrea Pedrini** · **Pietro Bellini**
