# StyleTimeQAM: Modeling Learning Style and Temporal Dynamics for Question-Answer Matching in Online Teaching Groups

This repository provides the implementation of **StyleTimeQAM** for **Teaching Group Question-Answer Matching (TGQAM)**.

## 1. Abstract

We introduce Teaching Group Question-Answer Matching (TGQAM), which aims to identify questions and their corresponding answers from mixed chronological conversations in online teaching groups. TGQAM faces three main challenges: the lack of domain-specific datasets, the presence of substantial irrelevant dialogue, and large candidate-answer spaces. To support this task, we collect two real-world datasets from two annual cohorts of the same university course and construct an auxiliary synthetic benchmark that preserves selected temporal and student-activity distributions. We propose StyleTimeQAM, which combines a Learning-Style-Aware Attention Module with a Time-Aware Attention Module. In this work, learning style refers to task-specific interaction tendencies rather than fixed cognitive categories. The former integrates learned student-specific representations with conversation content to improve question identification, while the latter models personalized temporal decay and relation information for answer matching. On BigData22 and BigData23, StyleTimeQAM achieves AUC scores of 0.8985/0.9353 under the TGQAM setting and 0.8729/0.9227 under the traditional QA setting, supporting the effectiveness of student-specific interaction and temporal modeling on the evaluated real teaching-group data.

## 2. Installation

Create and activate a Conda environment:

```bash
conda create -n STQAM_Env python=3
conda activate STQAM_Env
```

Install the required packages:

```bash
pip install torch
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install gensim
pip install tqdm
pip install tensorboardX
pip install json5
```

> Note: The exact Python version used in the original experimental environment should be specified here once confirmed.

## 3. Running

### 3.1 Dataset Selection

Three datasets are provided:

- `BigData22`
- `BigData23`
- `Synthetic`

![dataset_type](./Figures/dataset_statistics.png "dataset_statistics")

For StyleTimeQAM, the processed datasets are stored under:

```text
data/LEA_MODEL/
```

For example:

```text
bigdata22_train.csv
bigdata22_valid.csv
bigdata22_test.csv
```

Before running the baseline models, extract the following archives from:

```text
data/Baselines/
```

Files to extract:

```text
bigdata22_dataset.7z
bigdata23_dataset.7z
synthetic_dataset.7z
```

Place the extracted files into:

```text
data/baseline_data/
```

The baseline scripts read their input data from this directory.

The baseline models and StyleTimeQAM use different processed dataset formats. The processed baseline dataset contains three columns, while the processed StyleTimeQAM dataset contains five columns.

For StyleTimeQAM, the five columns are:

1. **sentence**  
   The input conversation sentence.

2. **user_ID**  
   The student who wrote the sentence.

3. **label**  
   Indicates whether the sentence is a question:
   - `1`: question
   - `0`: non-question

4. **match**  
   Represents the matching relationship between the current sentence and the following 100 candidate sentences:
   - `1`: the current sentence is a question and a future candidate sentence is its corresponding answer
   - `0`: mismatch

5. **timestamp**  
   Records when the sentence was written.

### 3.2 Running Baseline Models

We use `BigData22` and `CNN` as examples.

Run CNN under the **TGQAM setting**, where question labels are not explicitly provided:

```bash
python Baselines_main.py --dataset_type bigdata22 --model_type CNN --with_label 0
```

Run CNN under the **traditional QA setting**, where question labels are explicitly provided:

```bash
python Baselines_main.py --dataset_type bigdata22 --model_type CNN --with_label 1
```

### 3.3 Running StyleTimeQAM

Run StyleTimeQAM on `BigData22` under the **TGQAM setting**:

```bash
python STQAM_main.py --dataset_type bigdata22 --with_label 0
```

Run StyleTimeQAM on `BigData22` under the **traditional QA setting**, where question labels are explicitly provided:

```bash
python STQAM_main.py --dataset_type bigdata22 --with_label 1
```

## 4. Results

### 4.1 Experimental Results

![experiment_result](./Figures/experiment_result1.png "experiment_result1")

### 4.2 Baseline Hyperparameters

In our setting:

- `batch_size = 128`
- `max_length = 50`
- `dropout = 0.5`

#### TGQAM Setting

The following table shows the baseline configurations under the TGQAM setting:

| Model | BigData22 | BigData23 | Synthetic |
|---|---|---|---|
| AP-CNN | `wd`: 1e-5, `lr`: 1e-3 | `wd`: 5e-6, `lr`: 5e-4 | `wd`: 1e-5, `lr`: 1e-4 |
| BiLSTM-attention | `wd`: 1e-5, `lr`: 5e-4 | `wd`: 5e-6, `lr`: 5e-4 | `wd`: 1e-4, `lr`: 5e-3 |
| AP-LSTM | `wd`: 1e-5, `lr`: 5e-5 | `wd`: 1e-6, `lr`: 5e-3 | `wd`: 5e-5, `lr`: 1e-4 |
| CNN | `wd`: 5e-5, `lr`: 5e-4 | `wd`: 5e-6, `lr`: 1e-3 | `wd`: 1e-4, `lr`: 5e-5 |
| CNN-LSTM-CRF | `wd`: 1e-5, `lr`: 5e-3 | `wd`: 1e-6, `lr`: 5e-4 | `wd`: 1e-4, `lr`: 5e-3 |
| ABCNN | `wd`: 1e-6, `lr`: 1e-4 | `wd`: 5e-6, `lr`: 1e-3 | `wd`: 5e-5, `lr`: 5e-3 |
| ESIM | `wd`: 1e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 5e-4 | `wd`: 1e-4, `lr`: 5e-4 |

#### Traditional QA Setting

The following table shows the baseline configurations under the traditional QA setting:

| Model | BigData22 | BigData23 | Synthetic |
|---|---|---|---|
| AP-CNN | `wd`: 5e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 5e-4 |
| BiLSTM-attention | `wd`: 5e-5, `lr`: 1e-3 | `wd`: 1e-6, `lr`: 1e-3 | `wd`: 1e-5, `lr`: 1e-3 |
| AP-LSTM | `wd`: 5e-5, `lr`: 1e-4 | `wd`: 1e-6, `lr`: 1e-3 | `wd`: 1e-4, `lr`: 5e-3 |
| CNN | `wd`: 5e-5, `lr`: 5e-3 | `wd`: 5e-6, `lr`: 1e-3 | `wd`: 1e-5, `lr`: 1e-3 |
| CNN-LSTM-CRF | `wd`: 5e-5, `lr`: 1e-3 | `wd`: 5e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 5e-4 |
| ABCNN | `wd`: 1e-5, `lr`: 5e-5 | `wd`: 1e-6, `lr`: 5e-4 | `wd`: 1e-4, `lr`: 5e-4 |
| ESIM | `wd`: 1e-5, `lr`: 5e-3 | `wd`: 1e-5, `lr`: 5e-4 | `wd`: 1e-5, `lr`: 5e-5 |

### 4.3 Ablation Study

#### TGQAM Setting

![Ablation study under the TGQAM setting](./Figures/Ablation_TGQAM.png "Ablation study under the TGQAM setting")

#### Traditional QA Setting

![Ablation study under the traditional QA setting](./Figures/Ablation_Traditional.png "Ablation study under the traditional QA setting")

The ablation results on BigData22 under both TGQAM and traditional QA settings show that removing question extraction, temporal information, or student-specific interaction information leads to clear performance degradation. Removing the user relation matching module has a smaller effect because student-specific information is already incorporated into conversation representations by the Learning-Style-Aware Attention Module. These results support the usefulness of learned student-specific representations rather than predefined binary learning-style labels.

## 5. Citation

If you find this work useful, please cite our BESC 2026 paper.

The official BibTeX entry will be added after the Springer proceedings are published.
