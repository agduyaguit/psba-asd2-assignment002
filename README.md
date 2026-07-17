# Wearable-Sensor Classification of Daily and Sports Activities

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agduyaguit/psba-asd2-assignment002/blob/master/ASD2_A2_152QCQTQ.ipynb)

Deep-learning and machine-learning comparison for recognising **19 daily and sports activities** from multibody wearable-sensor recordings.


## Project information

- **Student:** Arnil Duyaguit
- **Student ID:** 152QCQTQ
- **Notebook:** `ASD2_A2_152QCQTQ.ipynb`
- **Environment:** Google Colab
- **Frameworks:** TensorFlow/Keras, XGBoost, scikit-learn
- **Dataset:** UCI Daily and Sports Activities

## Problem statement

The objective is to classify a five-second wearable-sensor recording into one of 19 activity classes.

Each observation contains:

- **125 timesteps**
- **45 sensor channels**
- **5 body locations:** torso, right arm, left arm, right leg, and left leg
- **3 sensor types per location:** accelerometer, gyroscope, and magnetometer
- **3 axes per sensor:** x, y, and z

The system is evaluated using a **participant-held-out split** to test whether it can recognise activities performed by people who were not included during model training.

## Dataset

The project uses the [UCI Daily and Sports Activities Dataset](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities).

The notebook downloads the dataset automatically from:

```text
https://archive.ics.uci.edu/static/public/256/daily%2Band%2Bsports%2Bactivities.zip
```

Verified dataset properties:

| Property | Value |
|---|---:|
| Total recordings | 9,120 |
| Activities | 19 |
| Participants | 8 |
| Recordings per activity | 480 |
| Recordings per participant | 1,140 |
| Sequence shape | 125 × 45 |
| Missing or infinite values | 0 |
| Exact duplicate sequences | 0 |

The activity labels are:

| Code | Activity |
|---|---|
| A01 | Sitting |
| A02 | Standing |
| A03 | Lying on back |
| A04 | Lying on right side |
| A05 | Ascending stairs |
| A06 | Descending stairs |
| A07 | Standing in a still elevator |
| A08 | Moving around in an elevator |
| A09 | Walking in a parking lot |
| A10 | Treadmill walking, flat |
| A11 | Treadmill walking, inclined |
| A12 | Treadmill running |
| A13 | Stepper exercise |
| A14 | Cross-trainer exercise |
| A15 | Exercise bike, horizontal |
| A16 | Exercise bike, vertical |
| A17 | Rowing |
| A18 | Jumping |
| A19 | Playing basketball |

## Data split

A participant-based split is used instead of a random recording-level split.

| Split | Participants | Samples | Samples per activity |
|---|---|---:|---:|
| Training | P1–P5 | 5,700 | 300 |
| Validation | P6 | 1,140 | 60 |
| Test | P7–P8 | 2,280 | 120 |

This design prevents recordings from the same person from appearing in both training and testing.

> The test-subject configuration should be `["p7", "p8"]`.

## Preprocessing

The notebook performs the following steps:

1. Downloads and extracts the raw UCI dataset.
2. Creates a metadata table from the activity, participant, and segment folders.
3. Verifies file count, sequence dimensions, missing values, infinite values, and duplicates.
4. Applies the participant-held-out split.
5. Calculates channel-wise mean and standard deviation using the **training set only**.
6. Applies the same normalisation statistics to validation and test data.
7. Creates efficient `tf.data` pipelines using caching, batching, shuffling for training, and prefetching.

Validation and test datasets are not shuffled so that predictions remain aligned with the original labels, participant identifiers, and error-analysis metadata.

## Models

### 1. XGBoost baseline

The baseline converts every sequence into **360 hand-crafted features**:

- mean
- standard deviation
- minimum
- maximum
- median
- range
- root mean square
- mean energy

Each statistic is calculated independently for all 45 channels.

### 2. One-dimensional CNN

The CNN contains:

- three `Conv1D` blocks
- batch normalisation
- ReLU activation
- max pooling
- global average pooling
- dropout
- L2 kernel regularisation
- a 19-class softmax output

The CNN sensitivity experiment varied:

- initial filters: `32` and `64`
- kernel size: `3` and `5`

The selected configuration was:

```text
Initial filters: 32
Kernel size: 5
Dropout: 0.30
Parameters: 52,627
```

### 3. Bidirectional LSTM

The BiLSTM contains:

- a bidirectional LSTM layer
- layer normalisation
- global average pooling
- dropout
- a dense hidden layer
- L2 kernel regularisation
- a 19-class softmax output

The BiLSTM sensitivity experiment varied:

- LSTM units: `32` and `64`
- dropout: `0.20` and `0.40`

The selected configuration was:

```text
LSTM units: 64
Dropout: 0.20
Parameters: 66,067
```

## Training configuration

Both deep-learning models use:

- Adam optimiser
- initial learning rate of `0.001`
- sparse categorical cross-entropy
- batch size of `64`
- maximum of `10` epochs in the executed experiment
- `ModelCheckpoint`
- `ReduceLROnPlateau`
- `EarlyStopping`
- fixed random seed of `42`

`ModelCheckpoint` preserves the lowest-validation-loss model. `ReduceLROnPlateau` halves the learning rate after three stagnant epochs, while `EarlyStopping` restores the best weights when validation loss stops improving.

## Results

### Validation results

| Model/configuration | Validation accuracy | Validation Macro-F1 |
|---|---:|---:|
| XGBoost | 0.8904 | 0.8833 |
| CNN: 32 filters, kernel 5 | 0.9509 | 0.9501 |
| BiLSTM: 64 units, dropout 0.20 | 0.9342 | 0.9311 |

### Held-out test results

| Model | Accuracy | Macro-F1 | Weighted F1 | Model size | Inference time |
|---|---:|---:|---:|---:|---:|
| XGBoost baseline | **0.8825** | **0.8807** | **0.8807** | 4.175 MB | 0.0456 ms/sample |
| 1D CNN | 0.7978 | 0.7799 | 0.7799 | **0.673 MB** | 0.2029 ms/sample |
| Bidirectional LSTM | 0.7956 | 0.7845 | 0.7845 | 0.809 MB | 0.3129 ms/sample |

The XGBoost baseline produced the strongest held-out test performance and the smallest validation-to-test performance drop. The deep-learning models achieved high validation scores but generalised less effectively to participants P7 and P8.

## Evaluation and explainability

The notebook generates:

- accuracy, Macro-F1, and weighted F1
- classification reports
- confusion matrices
- per-class precision, recall, and F1
- per-class F1 comparison
- misclassification tables
- five visualised CNN errors
- frequent confusion-pair analysis
- input-gradient temporal saliency
- sensor-channel and body-location importance
- model-size and inference-latency comparison

The saliency example uses a correctly classified Sitting sample and aggregates input-gradient importance across timesteps, channels, and body locations.

## Repository structure

```text
psba-asd2-assignment002/
├── ASD2_A2_152QCQTQ.ipynb
├── README.md
```

During notebook execution, the following working structure is created:

```text
content/ADS2_A2_WearableHAR/
├── data/
│   ├── raw/
│   │   ├── daily_sports_activities.zip
│   │   └── daily_sports_activities/
│   └── processed/
│       ├── metadata.csv
│       ├── split_manifest.csv
│       ├── daily_sports_arrays.npz
│       └── normalisation_statistics.npz
├── figures/
├── models/
└── results/
```

> In Google Colab, runtime storage is temporary. Copy important models, figures, and result files to Google Drive before resetting the runtime.

## How to reproduce the results

### Option 1: Google Colab

1. Click the **Open in Google Colab** badge at the top of this README.
2. Select a GPU runtime from **Runtime → Change runtime type**. A GPU is recommended for the CNN and BiLSTM experiments but is not required for the XGBoost baseline.
3. Run the notebook from top to bottom using **Runtime → Run all**.
4. The dataset will be downloaded and extracted automatically.
5. Confirm the following dataset checks:

```text
Number of sensor files: 9120
X shape: (9120, 125, 45)
y shape: (9120,)
subjects shape: (9120,)
NaN count: 0
Infinite count: 0
```

6. Review the generated models, figures, CSV reports, and configuration files inside the project working directory.

### Option 2: Local Python environment

Clone the repository:

```bash
git clone https://github.com/agduyaguit/psba-asd2-assignment002.git
cd psba-asd2-assignment002
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install the requirements
```bash
python install -r requirement.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Open the notebook:

```bash
jupyter notebook ASD2_A2_152QCQTQ.ipynb
```

The notebook was executed using:

```text
Python 3.12.13
TensorFlow 2.20.0
NumPy 2.0.2
Pandas 2.2.2
Matplotlib 3.10.0
Scikit-learn 1.6.1
XGBoost 3.3.0
```

## Generated artefacts

The notebook saves:

- trained XGBoost, CNN, and BiLSTM models
- hyperparameter experiment tables
- model-comparison results
- classification reports
- per-class metrics
- confusion-pair reports
- misclassification records
- saliency rankings
- architecture diagrams
- training curves
- confusion matrices
- deployment comparison
- environment and selected-model configuration files


## Main conclusion

For this participant-held-out experiment, the XGBoost baseline provided the strongest combination of accuracy, Macro-F1, stability, and inference speed. The result shows that greater architectural complexity does not automatically produce better generalisation, particularly when the number of participants is limited.

## Licence and data attribution

The dataset is distributed by the UCI Machine Learning Repository. Review the dataset page for its current licence and citation requirements:

- [UCI Daily and Sports Activities Dataset](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)

The dataset is downloaded at runtime and is not stored in this repository.
