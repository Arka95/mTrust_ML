Hi. This file will help you understand the structure of the code and sequence of execution.
Please excuse the badly formatted code/files which I wrote in my college days (parts of 2016 and 2018) :P
I will try to refactor it when i get the time. 

#------------------------------------------------------------------------------------------------------------------------------
Please refer below for details of the implementation
Reference Publication: https://link.springer.com/article/10.1007/s11277-020-07879-x

#------------------------------------------------------------------------------------------------------------------------------

PART 1 (android app) of Implementation: ( app contains both fuzzy and ML based methododlogies for calculating Trust)
https://github.com/Arka95/Targe

#------------------------------------------------------------------------------------------------------------------------------
ABOUT:

This is PART 2 of the project:
where we train and generate our ML model, which can predict Trust value of a user based on the frequency, intimacy and recency

#------------------------------------------------------------------------------------------------------------------------------
REQUIREMENTS:

please install tensorflow ( preferably CUDA/GPU version), scikitlearn, pandasql, pandas, numpy and keras in your environment before proceeding
#------------------------------------------------------------------------------------------------------------------------------
FOLDER STRUCTURE:

genrerated folder contains files which were generated after executing scripts from the dataset. 
This will also contain intermediate tables 

data folder contains the csv file which is our database holding call records from actual users


#------------------------------------------------------------------------------------------------------------------------------
STEPS:
1. run trust_data_extractor.ipynb to filter out redundant data from the data/call_log_database.csv file
   this will generated 4 files: call_log.csv containing main data without clutter, in_table.csv, out_table.csv, user_tendency.csv
   the tendency is calculated based on the in and out tables.

2. From here onwards, we will be using the 2 tables, call_log and user_tendency to generate a trustdb training data  for
   training our ML model. This data will be containing the to and from calling id, their frequency, recency and intimacy.
   This is unlabelled data as we have not ascertained the result(class) vaue i.e. the TRUST value in the dataset. We use 
   K-Means clustering algorithms to cluster related records together to determine the class(trust) of the records we have.
   run the main_trust_data_generation.ipynb to generate all discussed in step 2

3. From step 2, we have generated our tarining data for our model. Now we have to train our model based on this data and its labels
   run train.py to train a lightweight BackProp model based on normalized training data from trustdb.csv . 
   save this model in models. Also in the export_tf folder, we save an eqivalent lightweight weighted graph from this model, which
   can be used in tensorflow lite (for android mobiles) to compute trust of a person.

#------------------------------------------------------------------------------------------------------------------------------

4. To visualize statistics of data from our data run visualize_stats.ipynb. 
   
5. To check the social interaction graph of our network of participants as in the database, run social_graph_builder.ipynb. 
   This might take a while, and the generated png file of the graph is not human readable as the number of nodes are extremely 
   large, however the associations can be viewed in a text file
