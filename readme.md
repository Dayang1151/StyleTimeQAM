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
