# StyleTimeQAM: Modeling Learning Style and Temporal Dynamics for Question-Answer Matching in Online Teaching Groups
This repository provides the implementation of StyleTimeQAM for Teaching Group Question-Answer Matching (TGQAM).

# 1. Abstract 
We propose StyleTimeQAM for Teaching Group Question-Answer Matching (TGQAM), a QA matching problem that aims to align questions with correct answers in online teaching groups. TGQAM is useful for assessing teaching quality and understanding student engagement, but it faces three challenges: (i) the scarcity of relevant datasets, (ii) the prevalence of unrelated dialogue in teaching groups, and (iii) the wide range of potential answers for each question. To address the first challenge, we collect two one-year datasets from an anonymized university course and develop a synthetic dataset that mimics teaching-group dynamics, providing a foundational resource for TGQAM research. To tackle the second and third challenges, StyleTimeQAM incorporates both learning-style and temporal information through two modules: the Learning-Style-Aware Attention Module and the Time-Aware Attention Module. The former filters out irrelevant dialogue by modeling students' tendencies to ask or answer questions, while the latter uses a personalized time decay kernel function to reduce irrelevant candidate answers and improve question-answer matching accuracy. Experimental results show that StyleTimeQAM achieves AUC scores of 0.8985/0.9353 under the TGQAM setting and 0.8729/0.9227 under the traditional QA setting on BigData22/BigData23, confirming the effectiveness of incorporating learning-style and temporal information into QA matching.


# 2. Install the Requirements of Experiment

    conda create -n STQAM_Env python=3
    conda activate STQAM_Env
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
    
 # 3. Running
 ## 3.1 Datasets Selection
Select a dataset you want to include.
There are three datasets: BigData22, BigData23, and Synthetic.
![dataset_type](./Figures/dataset_statistics.png "dataset_statistics")

For StyleTimeQAM, the processed datasets are stored under `data/LEA_MODEL/`, such as `bigdata22_train.csv`, `bigdata22_valid.csv`, and `bigdata22_test.csv`.

Before running the baseline models, please extract `bigdata22_dataset.7z`, `bigdata23_dataset.7z`, and `synthetic_dataset.7z` from `data/Baselines/`, and put the extracted files into `data/baseline_data/`, because the baseline scripts read data from `data/baseline_data/`.

The dataset format of the baseline and StyleTimeQAM is different.

The dataset for the baseline has three columns, while the dataset for StyleTimeQAM has five columns:

The first column is sentence, which represents the input sentence.

The second column is user_ID, which represents the student who wrote the sentence.

The third column is label, which indicates whether the sentence is a question: 1 denotes a question and 0 denotes a non-question.

The fourth column is match, which represents the matching relationship between the current sentence and the following 100 candidate sentences.

The fifth column is timestamp, which records when the sentence was written.
1 represents a match (the current sentence is the question and the future sentence is the answer), while 0 represents a mismatch.


## 3.2 Running baselines model
We use bigdata22 as an example of a dataset, and CNN as an example of a type of model(with questions noise).

    python Baselines_main.py --dataset_type bigdata22 --model_type CNN --with_label 0

We use bigdata22 as an example of a dataset, and CNN as an example of a type of model(without questions noise).

    python Baselines_main.py --dataset_type bigdata22 --model_type CNN --with_label 1

## 3.3 Running StyleTimeQAM model
We use bigdata22 as an example of a dataset(with questions noise).

    python STQAM_main.py --dataset_type bigdata22 --with_label 0

We use bigdata22 as an example of a dataset(without questions noise).

    python STQAM_main.py --dataset_type bigdata22 --with_label 1

# 4. Results
## 4.1 Experiment Results

![experiment_result](./Figures/experiment_result1.png "experiment_result1")

## 4.2 Basic configurations about baselines

In our setting, the batch_size is 128, the max_length is 50, and the dropout is 0.5.

The first table shows the baseline configurations under the TGQAM setting:

||bigdata22|bigdata23|synthetic|
|---|---|---|---|
|AP-CNN |`wd`: 1e-5, `lr`: 1e-3 | `wd`: 5e-6, `lr`: 5e-4| `wd`: 1e-5, `lr`: 1e-4|
|BiLSTM-attention|`wd`: 1e-5, `lr`: 5e-4|`wd`: 5e-6, `lr`: 5e-4 |`wd`: 1e-4, `lr`: 5e-3 |
|AP-LSTM|`wd`: 1e-5, `lr`: 5e-5 |`wd`: 1e-6, `lr`: 5e-3 |`wd`: 5e-5, `lr`: 1e-4 | 
|CNN|`wd`: 5e-5, `lr`: 5e-4 |`wd`: 5e-6, `lr`: 1e-3 |  `wd`: 1e-4, `lr`:5e-5| 
|CNN-LSTM-CRF|`wd`: 1e-5, `lr`: 5e-3 | `wd`: 1e-6, `lr`: 5e-4| `wd`: 1e-4, `lr`: 5e-3|
|ABCNN|`wd`: 1e-6, `lr`: 1e-4 | `wd`: 5e-6, `lr`: 1e-3|`wd`: 5e-5, `lr`:5e-3 | 
|ESIM|`wd`: 1e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 5e-4|`wd`: 1e-4, `lr`: 5e-4|

The second table shows the baseline configurations under the traditional QA setting:

||bigdata22|bigdata23|synthetic|
|---|---|---|---|
|AP-CNN |`wd`: 5e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 1e-4| `wd`: 1e-5, `lr`: 5e-4|
|BiLSTM-attention|`wd`: 5e-5, `lr`: 1e-3|`wd`: 1e-6, `lr`: 1e-3 |`wd`: 1e-5, `lr`: 1e-3 |
|AP-LSTM|`wd`: 5e-5, `lr`: 1e-4 |`wd`: 1e-6, `lr`: 1e-3 |`wd`: 1e-4, `lr`: 5e-3 | 
|CNN|`wd`: 5e-5, `lr`: 5e-3 |`wd`: 5e-6, `lr`: 1e-3 | `wd`: 1e-5, `lr`:1e-3| 
|CNN-LSTM-CRF|`wd`: 5e-5, `lr`: 1e-3 | `wd`: 5e-5, `lr`: 1e-4| `wd`: 1e-5, `lr`: 5e-4|
|ABCNN|`wd`: 1e-5, `lr`: 5e-5 | `wd`: 1e-6, `lr`: 5e-4|`wd`: 1e-4, `lr`:5e-4 | 
|ESIM|`wd`: 1e-5, `lr`: 5e-3| `wd`: 1e-5, `lr`: 5e-4|`wd`: 1e-5, `lr`: 5e-5|

## 4.3 Ablation experiment

![Ablation study under the TGQAM setting](./Figures/Ablation_TGQAM.png "Ablation study under the TGQAM setting")

![Ablation study under the traditional QA setting](./Figures/Ablation_Traditional.png "Ablation study under the traditional QA setting")

The ablation results on BigData22 under both TGQAM and traditional QA settings show that removing question extraction, time information, or user learning-style information leads to clear performance degradation. In contrast, removing the user relation matching module has a smaller effect, since learning-style information has already been incorporated into the conversation representation through the style-aware attention module. Therefore, user relation matching complements the conversation relation matching process.
