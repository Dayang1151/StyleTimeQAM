# StyleTimeQAM: A Novel Approach to Teaching Group Question-Answer Matching in Online Education
This repository is the official implementation of the Teaching Group Question-Answering Matching (TGQAM).

# 1. Abstract 
This paper introduces a novel task in the field of education, Teaching Group Question-Answer Matching (TGQAM). 
This task is critical in online education systems for assessing teaching quality and identifying students' learning styles and engagement levels.
However, TGQAM encounters significant challenges, primarily the scarcity of public conversation datasets encapsulating the unique features of teaching group communities and 
the extreme noise in the data. To counter the first hurdle, we collected and made public a conversation dataset from a course at a Chinese university, 
spanning two years. The second challenge arises from the mingling of questions with other conversations and a vast pool of potential answers per question.
Consequently, we introduced StyleTimeQAM, a User Style-Aware and Time-Aware Question Answering Matching Model. 
Experimental results across three datasets reveal that StyleTimeQAM outperforms baseline models, underscoring the effectiveness of the user style-aware and time-aware attention modules.


# 2. Install the Requirments of Experiment

    conda create -n STQAM_Env python=3
    conda activate STQAM_Env
    pip install torch
    pip install pandas
    pip install numpy
    pip install scikit-learn
    pip install matplotlib
    pip install seaborn
    
    
 # 3. Running
 ## 3.1 Datasets Selection
Select a dataset you want to include.
There are four kinds of dataset
![dataset_type](./Figures/dataset_type.png "dataset_type")

`data/<dataset>/<dataset>_<kind>_<seed>.csv`

The dataset format of the baseline and STQAM is different.

The dataset for the baseline has three columns:

The dataset for the STQAM has five columns:

The first column is sentencen.Represents the entered sentence.

The second column is user_ID.Represents which student spoke this sentence.

The third column is label.Represents whether the sentence is a problem, 1 is a problem and 0 is not a problem.

The forth column is match.Represents the matching relationship of this sentence to 100 sentences in the future.
1 represents the match (this sentence is the question, the future sentence is the answer). 0 means mismatch


# 3.2 Running baselines model
We use bigdata22 as an example of a dataset.and CNN as an example of a type of model(with questions noise).

    python Baselines_main.py --dataset_type bigdata22 --model_type CNN --with_label 0

We use bigdata22 as an example of a dataset.and CNN as an example of a type of model(without questions noise).

    python Baselines_main.py --dataset_type bigdata22 --model_type CNN --with_label 1

# 3.2 Running STQAM model
We use bigdata22 as an example of a dataset(with questions noise).

    python STQAM_main.py --dataset_type bigdata22 --with_label 0

We use bigdata22 as an example of a dataset(without questions noise).

    python STQAM_main.py --dataset_type bigdata22 --with_label 1

# 4. Results
## 4.1 Experiment Results

![experiment_result](./Figures/experiment_result.png "experiment_result")

# 4.2 Basic configurations about baselines

In our setting,the batch_size is 128,the max_length is 50.the dropout is 0.5.

First table is with questions noise:

||bigdata22|bigdata23|synthetic|
|---|---|---|---|
|AP-CNN |`wd`: 1e-5, `lr`: 1e-3 | `wd`: 5e-6, `lr`: 5e-4| `wd`: 1e-5, `lr`: 1e-4|
|BILSTM|`wd`: 1e-5, `lr`: 5e-4|`wd`: 5e-6, `lr`: 5e-4 |`wd`: 1e-4, `lr`: 5e-3 |
|AP-BILSTM|`wd`: 1e-5, `lr`: 5e-5 |`wd`: 1e-6, `lr`: 5e-3 |`wd`: 5e-5, `lr`: 1e-4 | 
|CNN|`wd`: 5e-5, `lr`: 5e-4 |`wd`: 5e-6, `lr`: 1e-3 |  `wd`: 1e-4, `lr`:5e-5| 
|CNN-BILSTM|`wd`: 1e-5, `lr`: 5e-3 | `wd`: 1e-6, `lr`: 5e-4| `wd`: 1e-4, `lr`: 5e-3|
|ABCNN|`wd`: 1e-6, `lr`: 1e-4 | `wd`: 5e-6, `lr`: 1e-3|`wd`: 5e-5, `lr`:5e-3 | 
|ESIM|`wd`: 1e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 5e-4|`wd`: 1e-4, `lr`: 5e-4|

First table is without questions noise:

||bigdata22|bigdata23|synthetic|
|---|---|---|---|
|AP-CNN |`wd`: 5e-5, `lr`: 1e-4 | `wd`: 1e-5, `lr`: 1e-4| `wd`: 1e-5, `lr`: 5e-4|
|BILSTM|`wd`: 5e-5, `lr`: 1e-3|`wd`: 1e-6, `lr`: 1e-3 |`wd`: 1e-5, `lr`: 1e-3 |
|AP-BILSTM|`wd`: 5e-5, `lr`: 1e-4 |`wd`: 1e-6, `lr`: 1e-3 |`wd`: 1e-4, `lr`: 5e-3 | 
|CNN|`wd`: 5e-5, `lr`: 5e-3 |`wd`: 5e-6, `lr`: 1e-3 | `wd`: 1e-5, `lr`:1e-3| 
|CNN-BILSTM|`wd`: 5e-5, `lr`: 1e-3 | `wd`: 5e-5, `lr`: 1e-4| `wd`: 1e-5, `lr`: 5e-4|
|ABCNN|`wd`: 1e-5, `lr`: 5e-5 | `wd`: 1e-6, `lr`: 5e-4|`wd`: 1e-4, `lr`:5e-4 | 
|ESIM|`wd`: 1e-5, `lr`: 5e-3| `wd`: 1e-5, `lr`: 5e-4|`wd`: 1e-5, `lr`: 5e-5|

# 4.4 Ablation experiment

The experimental results reveal that the removal of question extraction, time-awareness, and user learning styles significantly affects model performance. Conversely, the existence or removal of the user relation matching module only slightly influences the model's effectiveness. This effect can be attributed to the fact that user learning style information is already embedded in the distributed representation of the conversation within the user style-aware attention module. Therefore, the conversation relation matching process takes into account user learning styles, and user relation matching simply complements this process. 
