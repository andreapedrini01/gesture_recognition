/*
 * ============================================================
 *  keyword_spotting.ino
 *  On-device Keyword Spotting  —  Assignment #2
 * ============================================================
 *  Target  : Arduino Nano 33 BLE Sense Lite  (nRF52840, Cortex-M4F, 64 MHz)
 *  Editor  : Arduino Cloud  (cloud.arduino.cc)
 *
 *  Pipeline:
 *    PDM mic ──► PCM ring-buffer ──► MFCC (CMSIS-DSP) ──► CNN (TFLite Micro) ──► Serial
 *
 * ============================================================
 *  SETUP IN ARDUINO CLOUD
 * ============================================================
 *  1. Open your Sketch on cloud.arduino.cc
 *  2. Add the model.h file:
 *       → click "+" next to the sketch tab → "New file" → paste the content
 *  3. Add the required libraries from the "Libraries" panel:
 *       → search "Harvard_TinyMLx"  → Add to Sketch
 *       (PDM and CMSIS-DSP are already included in the Mbed OS Nano Boards core)
 *  4. Select the board: Tools → Board → Arduino Mbed OS Nano Boards → Nano 33 BLE
 *  5. Compile and upload
 *
 * ============================================================
 *  MFCC PIPELINE  —  must be IDENTICAL to the training notebook!
 * ============================================================
 *  The notebook must replicate this exact pipeline:
 *    1. int16 PCM → float32 / 32768  (range [-1, 1])
 *    2. Pre-emphasis per-frame: y[i] = x[i] - 0.97 * x[i-1]
 *    3. Hamming window: w[i] = 0.54 - 0.46 * cos(2*pi*i / (N-1))
 *    4. RFFT N=256 (no normalization, same as np.fft.rfft)
 *    5. Power spectrum: |X[k]|^2
 *    6. Mel filterbank: 26 triangular filters, 300-8000 Hz, NO normalization
 *    7. Log: ln(mel_energy + 1e-9)
 *    8. DCT-II (no normalization): dct[k] = sum_m cos(pi*k*(2m+1)/(2*N_MEL)) * log_mel[m]
 *    9. Per-coefficient normalization: (mfcc - mean) / std
 *   10. Layout: [N_MFCC, NUM_FRAMES] (transposed, coeff-major)
 */

// ─────────────────────────────────────────────────────────────
//  Includes
// ─────────────────────────────────────────────────────────────
#include <PDM.h>
#include <arm_math.h>

// TensorFlow Lite Micro  (Harvard_TinyMLx library)
#include <TinyMLShield.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Model weights + MFCC parameters + normalization constants
#include "model.h"

// ─────────────────────────────────────────────────────────────
//  Derived constants
// ─────────────────────────────────────────────────────────────
#define N_FFT_BINS      (FRAME_LEN / 2 + 1)          // 129
#define NUM_FRAMES      ((CLIP_SAMPLES - FRAME_LEN) / HOP_LEN + 1)  // 124
static const int NUM_CLASSES = N_CLASSES;

// ─────────────────────────────────────────────────────────────
//  Audio buffer
// ─────────────────────────────────────────────────────────────
#define AUDIO_BUF_LEN   CLIP_SAMPLES    // 16000 samples = 1 s
#define PDM_BUF_LEN     256

static int16_t      audioBuf[AUDIO_BUF_LEN];
static int16_t      pdmBuf[PDM_BUF_LEN];
static volatile int samplesRead = 0;
static int          writeCursor = 0;

// ─────────────────────────────────────────────────────────────
//  MFCC working buffers
// ─────────────────────────────────────────────────────────────
static float32_t _sig    [FRAME_LEN];
static float32_t _wind   [FRAME_LEN];
static float32_t _fftOut [FRAME_LEN];
static float32_t _power  [N_FFT_BINS];
static float32_t _melE   [N_MEL];
static float32_t _logMel [N_MEL];

// Precomputed lookup tables
static float32_t hammingWin [FRAME_LEN];
static float32_t melFB      [N_MEL][N_FFT_BINS];
static float32_t dctMat     [N_MFCC][N_MEL];

// MFCC output → CNN input  [N_MFCC × NUM_FRAMES] (transposed)
static float32_t mfccMatrix [N_MFCC * NUM_FRAMES];

// CMSIS RFFT instance
static arm_rfft_fast_instance_f32 rfft;

// ─────────────────────────────────────────────────────────────
//  TFLite Micro
// ─────────────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE  (50 * 1024)
static uint8_t tensorArena[TENSOR_ARENA_SIZE];

namespace {
  tflite::ErrorReporter*    errorReporter = nullptr;
  const tflite::Model*      tflModel      = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*             tflInput      = nullptr;
  TfLiteTensor*             tflOutput     = nullptr;
}

// ─────────────────────────────────────────────────────────────
//  PDM callback
// ─────────────────────────────────────────────────────────────
void onPDMdata() {
  int bytesAvail = PDM.available();
  int bytesRead  = PDM.read(pdmBuf, bytesAvail);
  samplesRead    = bytesRead / 2;
}

// ─────────────────────────────────────────────────────────────
//  Hz ↔ Mel conversion
// ─────────────────────────────────────────────────────────────
static inline float hzToMel(float hz)  { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static inline float melToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

// ─────────────────────────────────────────────────────────────
//  Hamming window
// ─────────────────────────────────────────────────────────────
static void buildHammingWindow() {
  for (int i = 0; i < FRAME_LEN; i++)
    hammingWin[i] = 0.54f - 0.46f * cosf(TWO_PI * i / (FRAME_LEN - 1));
}

// ─────────────────────────────────────────────────────────────
//  Mel filterbank — triangular, NO normalization (raw weights)
// ─────────────────────────────────────────────────────────────
static void buildMelFilterbank() {
  float melLow  = hzToMel(MEL_LOW_HZ);
  float melHigh = hzToMel(MEL_HIGH_HZ);

  float melPts[N_MEL + 2];
  for (int i = 0; i < N_MEL + 2; i++)
    melPts[i] = melLow + (float)i * (melHigh - melLow) / (float)(N_MEL + 1);

  int binIdx[N_MEL + 2];
  for (int i = 0; i < N_MEL + 2; i++) {
    int bin   = (int)floorf((FRAME_LEN + 1) * melToHz(melPts[i]) / (float)SAMPLE_RATE);
    binIdx[i] = constrain(bin, 0, N_FFT_BINS - 1);
  }

  memset(melFB, 0, sizeof(melFB));
  for (int m = 0; m < N_MEL; m++) {
    int lo  = binIdx[m];
    int ctr = binIdx[m + 1];
    int hi  = binIdx[m + 2];
    for (int k = lo; k < ctr && ctr != lo; k++)
      melFB[m][k] = (float)(k - lo) / (float)(ctr - lo);
    for (int k = ctr; k <= hi && hi != ctr; k++)
      melFB[m][k] = (float)(hi - k) / (float)(hi - ctr);
  }
}

// ─────────────────────────────────────────────────────────────
//  DCT-II matrix — NO normalization (raw cosine)
// ─────────────────────────────────────────────────────────────
static void buildDCTMatrix() {
  for (int k = 0; k < N_MFCC; k++)
    for (int m = 0; m < N_MEL; m++)
      dctMat[k][m] = cosf(PI * (float)k * (2.0f * m + 1.0f) / (2.0f * N_MEL));
}

// ─────────────────────────────────────────────────────────────
//  MFCC for a single frame
// ─────────────────────────────────────────────────────────────
static void computeMFCCFrame(const int16_t* frameStart, float32_t* mfccOut) {

  // 1. int16 → float [-1, 1]
  static float32_t _pcm[FRAME_LEN];
  for (int i = 0; i < FRAME_LEN; i++)
    _pcm[i] = (float32_t)frameStart[i] / 32768.0f;

  // 2. Pre-emphasis (per-frame)
  _sig[0] = _pcm[0];
  for (int i = 1; i < FRAME_LEN; i++)
    _sig[i] = _pcm[i] - PRE_EMPHASIS * _pcm[i - 1];

  // 3. Hamming window
  arm_mult_f32(_sig, hammingWin, _wind, FRAME_LEN);

  // 4. RFFT N=256
  arm_rfft_fast_f32(&rfft, _wind, _fftOut, 0);

  // 5. Power spectrum |X[k]|²
  _power[0]              = _fftOut[0] * _fftOut[0];
  _power[N_FFT_BINS - 1] = _fftOut[1] * _fftOut[1];
  arm_cmplx_mag_squared_f32(&_fftOut[2], &_power[1], N_FFT_BINS - 2);

  // 6. Mel filterbank
  for (int m = 0; m < N_MEL; m++)
    arm_dot_prod_f32(melFB[m], _power, N_FFT_BINS, &_melE[m]);

  // 7. Log (natural)
  for (int m = 0; m < N_MEL; m++)
    _logMel[m] = logf(_melE[m] + 1e-9f);

  // 8. DCT-II
  for (int k = 0; k < N_MFCC; k++)
    arm_dot_prod_f32(dctMat[k], _logMel, N_MEL, &mfccOut[k]);
}

// ─────────────────────────────────────────────────────────────
//  Full MFCC matrix: normalize + transpose [N_MFCC × NUM_FRAMES]
// ─────────────────────────────────────────────────────────────
static void computeAllMFCCs() {
  float32_t frameMfcc[N_MFCC];
  memset(mfccMatrix, 0, sizeof(mfccMatrix));

  for (int f = 0; f < NUM_FRAMES; f++) {
    computeMFCCFrame(&audioBuf[f * HOP_LEN], frameMfcc);
    for (int c = 0; c < N_MFCC; c++) {
      float32_t norm = (frameMfcc[c] - NORM_MEAN[c]) / NORM_STD[c];
      mfccMatrix[c * NUM_FRAMES + f] = norm;
    }
  }
}

// ─────────────────────────────────────────────────────────────
//  Inference
// ─────────────────────────────────────────────────────────────
static int runInference() {

  // Copy into input tensor
  if (tflInput->type == kTfLiteFloat32) {
    for (int i = 0; i < N_MFCC * NUM_FRAMES; i++)
      tflInput->data.f[i] = mfccMatrix[i];
  } else if (tflInput->type == kTfLiteInt8) {
    float   scale = tflInput->params.scale;
    int32_t zp    = tflInput->params.zero_point;
    for (int i = 0; i < N_MFCC * NUM_FRAMES; i++) {
      int32_t q = (int32_t)roundf(mfccMatrix[i] / scale) + zp;
      tflInput->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Invoke() failed!");
    return -1;
  }

  // Read output
  float scores[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (tflOutput->type == kTfLiteFloat32)
      scores[i] = tflOutput->data.f[i];
    else if (tflOutput->type == kTfLiteInt8)
      scores[i] = (tflOutput->data.int8[i] - tflOutput->params.zero_point)
                  * tflOutput->params.scale;
    else
      scores[i] = 0.0f;
  }

  int   bestIdx   = 0;
  float bestScore = scores[0];
  for (int i = 1; i < NUM_CLASSES; i++)
    if (scores[i] > bestScore) { bestScore = scores[i]; bestIdx = i; }

  // Print
  Serial.print("[KWS] ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(CLASS_LABELS[i]);
    Serial.print(": ");
    Serial.print(scores[i], 3);
    if (i < NUM_CLASSES - 1) Serial.print("  |  ");
  }
  Serial.println();
  Serial.print("  --> Guess: ");
  Serial.print(CLASS_LABELS[bestIdx]);
  Serial.print("  (confidence = ");
  Serial.print(bestScore * 100.0f, 1);
  Serial.println(" %)");
  Serial.println("----------------------------------------");

  return bestIdx;
}

// ─────────────────────────────────────────────────────────────
//  setup()
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("======================================");
  Serial.println("  Keyword Spotting  -  Assignment #2  ");
  Serial.println("  Arduino Nano 33 BLE Sense Lite       ");
  Serial.println("======================================\n");

  // 1. MFCC tables
  Serial.print("  [1/4] MFCC tables ...        ");
  buildHammingWindow();
  buildMelFilterbank();
  buildDCTMatrix();
  Serial.println("OK");

  // 2. CMSIS RFFT
  Serial.print("  [2/4] CMSIS RFFT N=256 ...   ");
  arm_rfft_fast_init_f32(&rfft, FRAME_LEN);
  Serial.println("OK");

  // 3. TFLite Micro
  Serial.print("  [3/4] Loading model ...      ");
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("FAIL (schema mismatch)");
    while (1);
  }
  Serial.println("OK");

  Serial.print("  [3/4] AllocateTensors ...    ");
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterp(
      tflModel, resolver, tensorArena, TENSOR_ARENA_SIZE, errorReporter);
  interpreter = &staticInterp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FAIL  --> increase TENSOR_ARENA_SIZE!");
    while (1);
  }
  Serial.println("OK");

  tflInput  = interpreter->input(0);
  tflOutput = interpreter->output(0);

  Serial.print("       Input  shape: [");
  for (int i = 0; i < tflInput->dims->size; i++) {
    Serial.print(tflInput->dims->data[i]);
    if (i < tflInput->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  Serial.print("       Output shape: [");
  for (int i = 0; i < tflOutput->dims->size; i++) {
    Serial.print(tflOutput->dims->data[i]);
    if (i < tflOutput->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  Serial.print("       Arena used:   ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");

  // 4. PDM microphone
  Serial.print("  [4/4] Microphone PDM ...     ");
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("FAIL");
    while (1);
  }
  Serial.println("OK");

  Serial.println("\nReady, listening...");
  Serial.print("  Classes:      ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(CLASS_LABELS[i]);
    if (i < NUM_CLASSES - 1) Serial.print(" | ");
  }
  Serial.println();
  Serial.print("  MFCC:         ");
  Serial.print(NUM_FRAMES); Serial.print(" frames x ");
  Serial.print(N_MFCC);    Serial.println(" coefficients");
  Serial.println("========================================");
}

// ─────────────────────────────────────────────────────────────
//  loop()  —  sliding window (0.5 s overlap)
// ─────────────────────────────────────────────────────────────
void loop() {

  if (samplesRead > 0) {
    noInterrupts();
    int n = samplesRead;
    samplesRead = 0;
    interrupts();

    for (int i = 0; i < n; i++) {
      if (writeCursor < AUDIO_BUF_LEN)
        audioBuf[writeCursor++] = pdmBuf[i];
    }
  }

  if (writeCursor >= AUDIO_BUF_LEN) {
    computeAllMFCCs();
    runInference();

    // Slide by 0.5 s
    int slide = AUDIO_BUF_LEN / 2;
    memmove(audioBuf, &audioBuf[slide],
            (AUDIO_BUF_LEN - slide) * sizeof(int16_t));
    writeCursor = AUDIO_BUF_LEN - slide;
  }
}
