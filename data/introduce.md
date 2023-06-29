# Dataset introduce

# 1. BigData2022 and BigData2023

The BigData datasets are collected from a course entitled "Big Data Mindset and Analysis," taught at a Chinese university. These datasets stem from group records accumulated over the years 2022 and 2023.

The data format of the BigData datasets is depicted in next Figure. The datasets are stored in the .csv format, encompassing five distinct components: sentence, user\_ID, label, match, and timestamp. The specific data types for each component are detailed below:

Timestamp (integer type): This serves as a timestamp, specifying when a user's statement was made. For computational simplicity, the initial timestamp is set to 10000, representing the time of the first sentence's inception.

User_ID (integer type): This constitutes a unique identification number assigned to each user.

Sentence (string type): This conveys the textual content of a user's chat record.

Label (integer type): This category denotes the nature of a user's chat record. The labels 0, 1, and 2 correspond to noise, a question, and an answer, respectively.

Match (string type): This signifies the matching correlation between the record and the subsequent 100 records. If the current sentence is a question and there is an appropriate answer within the following 100 sentences, it is denoted as 1; otherwise, it is labeled 0.

![dataset_type](./Figures/dataset_type.png "dataset_type")

# 2.Synthetic Dataset

 The Synthetic Dataset was constructed based on the BigData2022 dataset, with each conversation having characteristics identical to those in the BigData2022 and BigData2023 datasets. This dataset comprises 61 virtual students. During the generation of the Synthetic dataset, we ensured that the characteristics of the dataset, such as student learning styles and the probability distribution of the time interval between questions and answers, mirrored those of the BigData2022 dataset. Additionally, the text information for the Synthetic dataset was sourced from the Microsoft Research WikiQA Corpus https://www.microsoft.com/en-us/download/details.aspx?id=52419.
