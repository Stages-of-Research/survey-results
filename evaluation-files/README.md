### Evaluation Files

The evaluation was done in two different ways: 
0) We received the data from limesurvey: statistic_survey_data.xls and survey_data.csv
1) First, incomplete survey results were removed
2) The data was labeled
3) The labeled data was analysed manually (see problems_solutions_analysis.xlsx). Concurrently, we removed data from women that did not study or work in Germany.
4) The other analyses (areas and topics analyses), that were independent from the labels, were processed with a python script (areas_topics_analyses.py)

Please note, that we used the term 'technical' instead of 'research'(as mentioned in our paper) in our evaluation.

The input for 3) are the labeled problems/solutions files: cultural_labeled.csv, personal_labeled.csv, technical_labeled.csv, work_labeled.csv
The input for 4) are the validated datasets of both groups: dataset_cs.csv and dataset_others.csv






### Glossary:

* statistic_survey_data.xls = statistical analysis of lime survey
* survey_data.csv = raw data we collected with lime survey

* areas_topics_analyses.py = analyses for correlations and calculation of most frequent areas/ topics.
* dataset_cs.csv = Collection of data of the computer science group that is an input for the analyses in eval2.py
* dataset_others.csv = Collection of data of the others group that is an input for the analyses in eval2.py

* problems_solutions_analysis.xlsx = analysis of labeled problems and solutions
* cultural_labeled.csv = results of our labeling process for the cultural area
* personal_labeled.csv = results of our labeling process for the personal area
* technical_labeled.csv = results of our labeling process for the technical area (in the paper referred as research area)
* work_labeled.csv = results of our labeling process for the work area
