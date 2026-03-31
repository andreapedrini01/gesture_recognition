#include "arduino_secrets.h"

/*
 * ============================================================
 *  On-Device Gesture Recognition â Arduino Nano 33 BLE Sense
 *  Model: MLP 36 -> 50 -> 15 -> 5  (ReLU, ReLU, Softmax)
 *  Classes: Punch | Flex | Shake | UpDown | Circle
 *  IMU: Arduino_LSM9DS1
 *  Features: mean, std, rms, min, max (time) + PSD (freq)
 *  Axis order: aX, aY, aZ, gX, gY, gZ
 *  Feature order: [means x6, stds x6, rms x6, mins x6, maxs x6, psd x6]
 * ============================================================
 */

#include <Arduino_LSM9DS1.h>
#include "model.h"

// âââ CONFIGURATION âââ
#define WINDOW_SIZE       20      // samples per window (must match training)
#define NUM_AXES          6       // aX, aY, aZ, gX, gY, gZ
#define NUM_FEATURES      36      // 6 features x 6 axes
#define THRESHOLD         0.7f    // minimum softmax confidence
#define ACCEL_THRESHOLD   2.5f    // g â motion trigger threshold
#define TRIGGER_COOLDOWN  500     // ms â minimum time between classifications

// âââ WINDOW BUFFER + TRIGGER STATE âââââ
float         window_data[WINDOW_SIZE][NUM_AXES];
int           sample_count           = 0;
bool          capturing              = false;
unsigned long last_classification_ms = 0;

// âââ HELPER: ReLU âââ
float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

// âââ HELPER: Softmax (in-place) âââââ
void softmax(float* x, int n) {
  float max_val = x[0];
  for (int i = 1; i < n; i++)
    if (x[i] > max_val) max_val = x[i];

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < n; i++)
    x[i] /= sum;
}

// âââ FEATURE EXTRACTION ââââ
/*
 * Replicates Python extract_features():
 *   compute_time_features() -> [means(6), stds(6), rms(6), mins(6), maxs(6)]
 *   compute_psd()           -> [psd(6)]
 *   concatenated            -> 36 values
 *
 * Axis column order in window_data: aX=0, aY=1, aZ=2, gX=3, gY=4, gZ=5
 */
void extract_features(float features[NUM_FEATURES]) {

  float means[NUM_AXES] = {0};
  float stds[NUM_AXES]  = {0};
  float rms[NUM_AXES]   = {0};
  float mins[NUM_AXES];
  float maxs[NUM_AXES];

  // Init min/max
  for (int a = 0; a < NUM_AXES; a++) {
    mins[a] =  1e9f;
    maxs[a] = -1e9f;
  }

  // --- Mean ---
  for (int i = 0; i < WINDOW_SIZE; i++)
    for (int a = 0; a < NUM_AXES; a++)
      means[a] += window_data[i][a];
  for (int a = 0; a < NUM_AXES; a++)
    means[a] /= WINDOW_SIZE;

  // --- Std, RMS, Min, Max ---
  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int a = 0; a < NUM_AXES; a++) {
      float v = window_data[i][a];
      float d = v - means[a];
      stds[a] += d * d;
      rms[a]  += v * v;
      if (v < mins[a]) mins[a] = v;
      if (v > maxs[a]) maxs[a] = v;
    }
  }
  for (int a = 0; a < NUM_AXES; a++) {
    stds[a] = sqrtf(stds[a] / WINDOW_SIZE);
    rms[a]  = sqrtf(rms[a]  / WINDOW_SIZE);
  }

  // --- PSD via DFT (mean of |FFT|^2 per axis) ---
  // Matches: np.mean(np.abs(np.fft.fft(window, axis=0))**2, axis=0)
  float psd[NUM_AXES] = {0};
  for (int a = 0; a < NUM_AXES; a++) {
    for (int k = 0; k < WINDOW_SIZE; k++) {
      float re = 0.0f, im = 0.0f;
      for (int n = 0; n < WINDOW_SIZE; n++) {
        float angle = -2.0f * M_PI * k * n / WINDOW_SIZE;
        re += window_data[n][a] * cosf(angle);
        im += window_data[n][a] * sinf(angle);
      }
      psd[a] += re * re + im * im;
    }
    psd[a] /= WINDOW_SIZE;
  }

  // --- Pack features in Python order ---
  // [means x6 | stds x6 | rms x6 | mins x6 | maxs x6 | psd x6]
  for (int a = 0; a < NUM_AXES; a++) features[0  + a] = means[a];
  for (int a = 0; a < NUM_AXES; a++) features[6  + a] = stds[a];
  for (int a = 0; a < NUM_AXES; a++) features[12 + a] = rms[a];
  for (int a = 0; a < NUM_AXES; a++) features[18 + a] = mins[a];
  for (int a = 0; a < NUM_AXES; a++) features[24 + a] = maxs[a];
  for (int a = 0; a < NUM_AXES; a++) features[30 + a] = psd[a];
}

// âââ Z-SCORE NORMALIZATION âââââ
void normalize_features(float features[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    float std = norm_stds[i];
    features[i] = (std == 0.0f) ? 0.0f : (features[i] - norm_means[i]) / std;
  }
}

// âââ MLP FORWARD PASS ââââ
void mlp_forward(float* input, float* output) {
  float h0[50];
  float h1[15];

  // Layer 0: 36 -> 50 (ReLU)
  for (int j = 0; j < 50; j++) {
    float acc = biases_0[j];
    for (int i = 0; i < 36; i++)
      acc += input[i] * weights_0[i * 50 + j];
    h0[j] = relu(acc);
  }

  // Layer 1: 50 -> 15 (ReLU)
  for (int j = 0; j < 15; j++) {
    float acc = biases_1[j];
    for (int i = 0; i < 50; i++)
      acc += h0[i] * weights_1[i * 15 + j];
    h1[j] = relu(acc);
  }

  // Layer 2: 15 -> 5 (Softmax)
  for (int j = 0; j < NUM_CLASSES; j++) {
    float acc = biases_2[j];
    for (int i = 0; i < 15; i++)
      acc += h1[i] * weights_2[i * NUM_CLASSES + j];
    output[j] = acc;
  }

  softmax(output, NUM_CLASSES);
}

// âââ GESTURE CLASSIFICATION âââ
void classify_gesture() {
  float features[NUM_FEATURES];
  float probs[NUM_CLASSES];

  extract_features(features);
  normalize_features(features);
  mlp_forward(features, probs);

  // Find best class
  int   best_class = 0;
  float best_prob  = probs[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (probs[i] > best_prob) {
      best_prob  = probs[i];
      best_class = i;
    }
  }

  // Print result
  Serial.println("-----------------------------");
  if (best_prob >= THRESHOLD) {
    Serial.print("Gesture    : ");
    Serial.println(gesture_names[best_class]);
    Serial.print("Confidence : ");
    Serial.print(best_prob * 100.0f, 1);
    Serial.println("%");
  } else {
    Serial.println("Gesture    : UNKNOWN (low confidence)");
    Serial.print("Best guess : ");
    Serial.print(gesture_names[best_class]);
    Serial.print(" @ ");
    Serial.print(best_prob * 100.0f, 1);
    Serial.println("%");
  }

  // All class probabilities (debug)
  Serial.println("Probabilities:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print("  ");
    Serial.print(gesture_names[i]);
    Serial.print(": ");
    Serial.print(probs[i] * 100.0f, 1);
    Serial.println("%");
  }
  Serial.println("-----------------------------");
}

// âââ SETUP ââ
void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed!");
    while (1);
  }

  Serial.println("=== Gesture Recognition Ready ===");
  Serial.print("Accel sample rate : ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyro  sample rate : ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  Serial.print("Window size       : ");
  Serial.print(WINDOW_SIZE);
  Serial.println(" samples");
  Serial.print("Motion threshold  : ");
  Serial.print(ACCEL_THRESHOLD);
  Serial.println(" g");
  Serial.println("Waiting for motion...");
}

// âââ MAIN LOOP ââââââââââââââââââââââââââââââââââââââââââââââââ
void loop() {
  float ax, ay, az, gx, gy, gz;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // ââ wait for motion trigger ââ
    if (!capturing) {
      float mag        = sqrtf(ax*ax + ay*ay + az*az);
      unsigned long now = millis();
      bool cooldown_ok  = (now - last_classification_ms) > TRIGGER_COOLDOWN;

      if (mag > ACCEL_THRESHOLD && cooldown_ok) {
        capturing    = true;
        sample_count = 0;
        Serial.println(">> Motion detected â capturing window...");
      }
      return;
    }

    // ââ CAPTURING: fill window buffer ââ
    window_data[sample_count][0] = ax;
    window_data[sample_count][1] = ay;
    window_data[sample_count][2] = az;
    window_data[sample_count][3] = gx;
    window_data[sample_count][4] = gy;
    window_data[sample_count][5] = gz;
    sample_count++;

    // ââ Window full â classify ââ
    if (sample_count >= WINDOW_SIZE) {
      classify_gesture();
      last_classification_ms = millis();
      capturing    = false;
      sample_count = 0;
      Serial.println("Waiting for motion...");
    }
  }
}
