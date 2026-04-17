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
 *       → search "Arduino_TensorFlowLite"  → Add to Sketch
 *       (PDM and CMSIS-DSP are already included in the Mbed OS Nano Boards core)
 *  4. Select the board: Tools → Board → Arduino Mbed OS Nano Boards → Nano 33 BLE
 *  5. Compile and upload
 *
 *  How your partner should generate model.h (in Google Colab):
 *    !echo "const unsigned char model_data[] = {" > model.h
 *    !cat keyword_model.tflite | xxd -i               >> model.h
 *    !echo "};"                                        >> model.h
 *    → Download model.h and paste its content into the Arduino Cloud tab
 *
 * ============================================================
 *  MFCC PARAMETERS — must be IDENTICAL to the training Colab!
 * ============================================================
 */

// ─────────────────────────────────────────────────────────────
//  Includes
// ─────────────────────────────────────────────────────────────
#include <PDM.h>

// CMSIS-DSP
#include <arm_math.h>

// TensorFlow Lite Micro
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Weights of the Colab model
#include "model.h"   // defines: const unsigned char model_data[];

// ─────────────────────────────────────────────────────────────
//  MFCC Parameters
// ─────────────────────────────────────────────────────────────
#define SAMPLE_RATE     16000       // Hz  (Nyquist → 8 kHz, covers all speech)
#define FRAME_LEN       256         // samples per frame  = 16 ms  @ 16 kHz
#define HOP_LEN         128         // hop between frames  =  8 ms  (50% overlap)
#define N_FFT_BINS      (FRAME_LEN / 2 + 1)   // 129 unique bins from RFFT
#define N_MEL           26          // triangular Mel filters
#define N_MFCC          13          // cepstral coefficients to keep
#define MEL_LOW_HZ      300.0f      // minimum Mel filter frequency (Hz)
#define MEL_HIGH_HZ     8000.0f     // maximum Mel filter frequency (Hz)
#define PRE_EMPHASIS    0.97f       // pre-emphasis coefficient α

// Number of MFCC frames in the entire 1-second window
//   = (16000 − 256) / 128 + 1  ≈  123 frames
#define NUM_FRAMES      ((SAMPLE_RATE - FRAME_LEN) / HOP_LEN + 1)

// ─────────────────────────────────────────────────────────────
//  Classes
// ─────────────────────────────────────────────────────────────
static const char* CLASS_LABELS[] = { "clap", "tap", "snap", "silence" };
static const int   NUM_CLASSES    = 4;

// ─────────────────────────────────────────────────────────────
//  Audio buffer
// ─────────────────────────────────────────────────────────────
#define AUDIO_BUF_LEN   SAMPLE_RATE   // 1 s = 16 000 samples int16
#define PDM_BUF_LEN     256           // PDM callback buffer size

static int16_t          audioBuf[AUDIO_BUF_LEN];  // sliding window 1 s
static int16_t          pdmBuf[PDM_BUF_LEN];      // scratch for PDM driver
static volatile int     samplesRead = 0;
static          int     writeCursor = 0;

// ─────────────────────────────────────────────────────────────
//  MFCC working buffers  (all float32)
// ─────────────────────────────────────────────────────────────
static float32_t _sig    [FRAME_LEN];          // after pre-emphasis
static float32_t _wind   [FRAME_LEN];          // after Hamming window
static float32_t _fftOut [FRAME_LEN];          // RFFT output (Re/Im interleaved)
static float32_t _power  [N_FFT_BINS];         // power spectrum P[k]
static float32_t _melE   [N_MEL];              // Mel filterbank energies
static float32_t _logMel [N_MEL];              // log Mel energies

// Precomputed lookup tables
static float32_t hammingWin [FRAME_LEN];
static float32_t melFB      [N_MEL][N_FFT_BINS]; // [filter × FFT bin]
static float32_t dctMat     [N_MFCC][N_MEL];     // DCT-II matrix

// MFCC matrix → CNN input
static float32_t mfccMatrix [NUM_FRAMES * N_MFCC];  // [frame × coeff]

// CMSIS-DSP instance for real RFFT N=256
static arm_rfft_fast_instance_f32 rfft;

// ─────────────────────────────────────────────────────────────
//  TFLite Micro
// ─────────────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE  (50 * 1024)   // 50 KB
static uint8_t tensorArena[TENSOR_ARENA_SIZE];

// Namespace to avoid conflicts with local variables
namespace {
  tflite::ErrorReporter*   errorReporter = nullptr;
  const tflite::Model*     tflModel      = nullptr;
  tflite::MicroInterpreter* interpreter  = nullptr;
  TfLiteTensor*            tflInput      = nullptr;
  TfLiteTensor*            tflOutput     = nullptr;
}

// ─────────────────────────────────────────────────────────────
//  PDM interrupt callback
// ─────────────────────────────────────────────────────────────
void onPDMdata() {
  int bytesAvail = PDM.available();
  int bytesRead  = PDM.read(pdmBuf, bytesAvail);
  samplesRead    = bytesRead / 2;   // int16 → 2 bytes per sample
}

// ─────────────────────────────────────────────────────────────
//  Helper functions: conversion Hz ↔ Mel
// ─────────────────────────────────────────────────────────────
static inline float hzToMel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static inline float melToHz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// ─────────────────────────────────────────────────────────────
//  Precompute: Hamming window
//    w[i] = 0.54 − 0.46 · cos(2π·i / (N−1))
// ─────────────────────────────────────────────────────────────
static void buildHammingWindow() {
  for (int i = 0; i < FRAME_LEN; i++) {
    hammingWin[i] = 0.54f - 0.46f * cosf(TWO_PI * i / (FRAME_LEN - 1));
  }
}

// ─────────────────────────────────────────────────────────────
//  Precompute: Mel triangular filterbank
// ─────────────────────────────────────────────────────────────
static void buildMelFilterbank() {
  float melLow  = hzToMel(MEL_LOW_HZ);
  float melHigh = hzToMel(MEL_HIGH_HZ);

  // N_MEL + 2 equally spaced points on the Mel scale
  float melPts[N_MEL + 2];
  for (int i = 0; i < N_MEL + 2; i++) {
    melPts[i] = melLow + (float)i * (melHigh - melLow) / (float)(N_MEL + 1);
  }

  // Convert Mel → Hz → nearest FFT bin index
  int binIdx[N_MEL + 2];
  for (int i = 0; i < N_MEL + 2; i++) {
    int bin   = (int)floorf((FRAME_LEN + 1) * melToHz(melPts[i]) / (float)SAMPLE_RATE);
    binIdx[i] = constrain(bin, 0, N_FFT_BINS - 1);
  }

  // Triangular weights
  memset(melFB, 0, sizeof(melFB));
  for (int m = 0; m < N_MEL; m++) {
    int lo  = binIdx[m];
    int ctr = binIdx[m + 1];
    int hi  = binIdx[m + 2];

    for (int k = lo; k < ctr && ctr != lo; k++)
      melFB[m][k] = (float)(k - lo) / (float)(ctr - lo);   // rising slope
    for (int k = ctr; k <= hi && hi != ctr; k++)
      melFB[m][k] = (float)(hi - k) / (float)(hi - ctr);   // falling slope
  }
}

// ─────────────────────────────────────────────────────────────
//  Precompute: DCT-II orthogonal matrix
//    dctMat[k][m] = cos( π·k·(2m+1) / (2·N_MEL) )
// ─────────────────────────────────────────────────────────────
static void buildDCTMatrix() {
  for (int k = 0; k < N_MFCC; k++)
    for (int m = 0; m < N_MEL; m++)
      dctMat[k][m] = cosf(PI * (float)k * (2.0f * m + 1.0f) / (2.0f * N_MEL));
}

// ─────────────────────────────────────────────────────────────
//  Compute MFCC for a single frame
//    frameStart → int16_t[FRAME_LEN]   (PCM samples)
//    mfccOut    → float32_t[N_MFCC]    (output coefficients)
// ─────────────────────────────────────────────────────────────
static void computeMFCCFrame(const int16_t* frameStart, float32_t* mfccOut) {

  // Step 1: Pre-emphasis — enhance high frequencies
  //   sig[i] = frame[i] − α · frame[i−1]
  _sig[0] = (float32_t)frameStart[0];
  for (int i = 1; i < FRAME_LEN; i++)
    _sig[i] = (float32_t)frameStart[i] - PRE_EMPHASIS * (float32_t)frameStart[i-1];

  // Step 2: Hamming window — reduce spectral leakage at frame edges
  //   CMSIS arm_mult_f32: vectorized multiply with SIMD → 4 products per cycle
  arm_mult_f32(_sig, hammingWin, _wind, FRAME_LEN);

  // Step 3: RFFT real (N=256) → 129 unique complex bins
  //   CMSIS: [Re(0), Re(N/2), Re(1),Im(1), Re(2),Im(2), ...]
  arm_rfft_fast_f32(&rfft, _wind, _fftOut, 0);

  // Step 4: Power Spectrum  P[k] = Re²[k] + Im²[k]
  //   Bin DC (0) and Nyquist (128): Im = 0 by definition → only Re²
  _power[0]            = _fftOut[0] * _fftOut[0];              // DC
  _power[N_FFT_BINS-1] = _fftOut[1] * _fftOut[1];              // Nyquist
  //   Bins 1..127: Re/Im pairs interleaved from _fftOut[2]
  arm_cmplx_mag_squared_f32(&_fftOut[2], &_power[1], N_FFT_BINS - 2);

  // Step 5: Mel filterbank
  for (int m = 0; m < N_MEL; m++)
    arm_dot_prod_f32(melFB[m], _power, N_FFT_BINS, &_melE[m]);

  // Step 6: Logarithm
  for (int m = 0; m < N_MEL; m++)
    _logMel[m] = logf(_melE[m] + 1e-9f);

  // Step 7: DCT-II → N_MFCC
  for (int k = 0; k < N_MFCC; k++)
    arm_dot_prod_f32(dctMat[k], _logMel, N_MEL, &mfccOut[k]);
}

// ─────────────────────────────────────────────────────────────
// Compute the full MFCC matrix over the 1-second window
// Fills mfccMatrix[NUM_FRAMES × N_MFCC]
// ─────────────────────────────────────────────────────────────
static void computeAllMFCCs() {
  for (int f = 0; f < NUM_FRAMES; f++)
    computeMFCCFrame(&audioBuf[f * HOP_LEN], &mfccMatrix[f * N_MFCC]);
}

// ─────────────────────────────────────────────────────────────
// TFLite Micro inference + print result on Serial
// ─────────────────────────────────────────────────────────────
static int runInference() {

  // ── Copy MFCC features into input tensor ────────────
  if (tflInput->type == kTfLiteFloat32) {
    // float32 model
    for (int i = 0; i < NUM_FRAMES * N_MFCC; i++)
      tflInput->data.f[i] = mfccMatrix[i];

  } else if (tflInput->type == kTfLiteInt8) {
    // int8 quantized model: q = float / scale + zero_point
    float   scale  = tflInput->params.scale;
    int32_t zp     = tflInput->params.zero_point;
    for (int i = 0; i < NUM_FRAMES * N_MFCC; i++) {
      int32_t q = (int32_t)roundf(mfccMatrix[i] / scale) + zp;
      tflInput->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }
  } else {
    Serial.println("[ERROR] Unsupported input tensor type!");
    return -1;
  }

  // ── Run model ─────────────────────────────────────
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Invoke() failed!");
    return -1;
  }

  // ── Read output probabilities ─────────────────────
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

  // Find class with highest probability
  int   bestIdx   = 0;
  float bestScore = scores[0];
  for (int i = 1; i < NUM_CLASSES; i++)
    if (scores[i] > bestScore) { bestScore = scores[i]; bestIdx = i; }

  // ── Print scores ────────────────────────────────
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
  while (!Serial);   // wait for Serial Monitor to open

  Serial.println("======================================");
  Serial.println("  Keyword Spotting  -  Assignment #2  ");
  Serial.println("  Arduino Nano 33 BLE Sense Lite       ");
  Serial.println("======================================");
  Serial.println();

  // ── 1. Precompute MFCC tables ───────────────────────────
  Serial.print("  [1/4] Hamming window ...  ");
  buildHammingWindow();
  Serial.println("OK");

  Serial.print("  [1/4] Mel filterbank  ...  ");
  buildMelFilterbank();
  Serial.println("OK");

  Serial.print("  [1/4] DCT-II matrix   ...  ");
  buildDCTMatrix();
  Serial.println("OK");

  // ── 2. Initialize CMSIS RFFT (N=256) ───────────────────
  Serial.print("  [2/4] CMSIS RFFT N=256 ... ");
  arm_rfft_fast_init_f32(&rfft, FRAME_LEN);
  Serial.println("OK");

  // ── 3. Initialize TFLite Micro ──────────────────────────
  Serial.print("  [3/4] Loading model ...      ");
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  tflModel = tflite::GetModel(model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("FAIL  (schema: model=");
    Serial.print(tflModel->version());
    Serial.print(" expected=");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println(")");
    while (1);
  }
  Serial.println("OK");

  Serial.print("  [3/4] AllocateTensors ...   ");
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

  // Debug: tensor shapes
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
  Serial.print("       Arena used:  ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");

  // ── 4. Start PDM microphone ───────────────────────────────
  Serial.print("  [4/4] Microphone PDM ...     ");
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1 /* mono */, SAMPLE_RATE)) {
    Serial.println("FAIL  --> PDM.begin() returned false!");
    while (1);
  }
  Serial.println("OK");

  // Configuration summary
  Serial.println();
  Serial.println("Ready, listening...");
  Serial.print("  Classes:        ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(CLASS_LABELS[i]);
    if (i < NUM_CLASSES - 1) Serial.print(" | ");
  }
  Serial.println();
  Serial.print("  MFCC frames:   "); Serial.print(NUM_FRAMES);
  Serial.print(" frames x "); Serial.print(N_MFCC); Serial.println(" coefficients");
  Serial.print("  Window:        "); Serial.print(AUDIO_BUF_LEN); Serial.println(" samples (1 s)");
  Serial.println("========================================");
}

// ─────────────────────────────────────────────────────────────
//  loop()
//
//  Sliding-window strategy for continuous detection:
//    1. Fills audioBuf with 1 second of PCM data.
//    2. Computes MFCC → runs inference → prints result on Serial.
//    3. Slides the window by 0.5 s (keeps the last half
//       as context for the next inference).
// ─────────────────────────────────────────────────────────────
void loop() {

  // Drain the PDM buffer (filled by the ISR callback) into audioBuf
  if (samplesRead > 0) {
    noInterrupts();
    int n     = samplesRead;
    samplesRead = 0;
    interrupts();

    for (int i = 0; i < n; i++) {
      if (writeCursor < AUDIO_BUF_LEN)
        audioBuf[writeCursor++] = pdmBuf[i];
    }
  }

  // When we have accumulated a full 1-second window → MFCC + inference
  if (writeCursor >= AUDIO_BUF_LEN) {

    computeAllMFCCs();   // compute MFCC matrix [NUM_FRAMES x N_MFCC]
    runInference();      // CNN inference + print output

    // Sliding window: discard the first 0.5 s, keep the last 0.5 s
    int slide = AUDIO_BUF_LEN / 2;
    memmove(audioBuf, &audioBuf[slide],
            (AUDIO_BUF_LEN - slide) * sizeof(int16_t));
    writeCursor = AUDIO_BUF_LEN - slide;
  }
}
