

import pandas as pd
from IPython.display import display
import numpy as np
from pathlib import Path
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Part Two - Feature Extraction and Classification

# In the second part of the coursework, your task is to train and test machine learning classifiers on a dataset of
# political speeches. The objective is to learn to predict the political party from the text of the speech. The texts you
# need for this part are in the speeches subdirectory of the texts directory of the coursework. Moodle template.
# For this part, you can structure your python functions in any way that you like, but pay attention to exactly what
# information (if any) you are asked to print out in each part. Your final scripts should print out the answers to each
# part where required, and nothing else


def read_speeches(path=Path.cwd() / "texts" / "speeches"):
    """Reads speeches (csv text file) located as a specified path and does some preprocessing and returns dimensions of
       pandas datafrome containing the speeches data
       Requires a path to folder location where the speech text CSV file is located.

        Args:
            path (str): path location to the folder containing speeches text files to preprocess.
        Returns:
            pandas dataframe: pandas dataframe containing pre-processed speeches data
    """

    # (a) Read the handsard40000.csv dataset in the texts directory into a dataframe. Subset and rename the dataframe
    # as follows;
    # REF https://pandas.pydata.org/docs/reference/api/pandas.set_option.html
    df = pd.read_csv(path/"hansard40000.csv")
    # print(df)  #PLEASE UNCOMMENT IF YOU WOULD LIKE TO DISPLAY DATAFRAME


    pd.set_option('display.max_columns', None)      # Display all columns. None - unlimited
    pd.set_option('display.max_rows', None)         # Display all rows. None - unlimited
    pd.set_option('display.width', None)            # Display width in characters for pandas. None - auto detects width


    # (i) rename the 'Labour (Co-op)' value in 'party' column to 'Labour', and then:   [DONE]
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    # display(df) #PLEASE UNCOMMENT IF YOU WOULD LIKE TO DISPLAY DATAFRAME

    # (ii) remove any rows where the value of the 'party' column is not one of the four most common party names and
    # remove the 'Speaker' value. [DONE]

    # Get the frequency counts of each party
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
    party_counts = df['party'].value_counts()     # Descending order returns list of counts for each party
    # display(party_counts)  #PLEASE UNCOMMENT IF YOU WOULD LIKE TO DISPLAY DATAFRAME

    # Get the names of the 4 most common parties from list  [DONE]
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
    #REF https://en.moonbooks.org/Articles/How-to-extract-the-value-names-and-counts-from-valuecounts-in-pandas-/

    party_names_list = df['party'].value_counts().index.tolist()
    # print("The top 4 common parties are " + str(party_names_list[0] + "," + str(party_names_list[1] + \
    #                            "," + str(party_names_list[2] ) + "," + str(party_names_list[4] ))))
    # Code above used to determine the 4 common parties to then remove rows from dataframe in the next step that
    # are not those parties or the Speaker party

    # Now can remove the records from dataframe that contain non-common parties and also where party is Speaker [DONE]
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
    #REF https://www.geeksforgeeks.org/drop-rows-from-the-dataframe-based-on-certain-condition-applied-on-a-column/

    df = df.drop(df[(df['party'] == 'Democratic Unionist Party') | (df['party'] == 'Independent')  | \
                (df['party'] == 'Plaid Cymru') | \
                (df['party'] == 'Social Democratic & Labour Party') | (df['party'] == 'Alliance') | \
                (df['party'] == 'Green Party')  |  (df['party'] == 'Speaker') | \
                (df['party'] == 'Alba Party')].index)


    # (iii) remove any rows where the value in the 'speech_class' column is not 'Speech'. [DONE]
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
    #REF https://www.geeksforgeeks.org/drop-rows-from-the-dataframe-based-on-certain-condition-applied-on-a-column/
    df = df.drop(df[(df['speech_class'] != 'Speech')].index)
    #display(df) - PLEASE UNCOMMENT FOR DISPLAY/DEBUG

    # (iv) remove any rows where the text in the 'speech' column is less than 1000 characters long [DONE]
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
    #REF https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html

    #For QA for (iv)
    #print(df.speech.str.len())  - PLEASE UNCOMMENT IF YOU WOULD LIKE TO CHECK LENGTHS OF SPEECH TEXT

    df = df.drop(df[((df.speech.str.len() < 1000))].index)
    #display(df) - PLEASE UNCOMMENT FOR DISPLAY/DEBUG


    # Print the dimensions of the resulting dataframe using the shape method  [DONE]
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
    print(df.shape)
    return df



def LoadData_and_ExtractFeatures(df):
    # (b) vectorise the speeches using TfidVectorizer from scikit-learn. Use the default parameters, except for
    # omitting English stopwords and setting max_features to 3000. Split the data into a train and test set, using
    # stratified sampling, with a random seed of 26      [DONE]
    # REF https://code.likeagirl.io/good-train-test-split-an-approach-to-better-accuracy-91427584b614 for the types of
    # train test splits
    # REF https://builtin.com/data-science/train-test-split#:~:text=Stratified%20Splitting&text=This%20creates%20training%20and%20testing,categories%20aren't%20represented%20equally.
    # for what stratified sampling means
    # REF https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # for parameters and their defaults and what max_features and stratified means

    """Load data

        Args:
             df: pandas dataframe containing x (data) and y (labels)
        Returns:
             none
    """
    # Checking the columns exist in the dataframe
    # display(df.columns) #FOR DEBUG

    #Split the dataframe data intothe required  x (data) and y (target labels)
    X = df['speech']
    y = df['party']

    #display(X)  # FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO iNSPECT DATA ('speech)
    #display(y)  # FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO iNSPECT LABELS DATA ('party')

    # Split the X (speeches) and y (party labels) data into training set 75% and 25% for testing set
    # using stratifIed sampling, max_features set to 3000,
    # Where: max features means, if not None, is used to build a vocabulary that considers the top max_features
    # ordered by term frequency across the corpus. Otherwise, all features are used (taken from
    # TfidVectorizer scikit learn online help pages)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=26, stratify=y)

    #REF https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html for examples
    # of extracting features for train and test data using TfIdfVectorizer

    # Extracting features using a sparse vectorizer TfIdf
    t0 = time()
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )
    X_train_extracted_features = vectorizer.fit_transform(X_train)
    duration_train = time() - t0

    # Extracting features from the test data using the same vectorizer
    t0 = time()
    X_test_extracted_features = vectorizer.transform(X_test)
    duration_test = time() - t0

    #print(f"{len(X_train)} documents (training set)")   # FOR DEBUG
    #print(f"{len(X_test)} documents (testing set)")     # FOR DEBUG
    #print("Vectorise training done: %f" % duration_train ," seconds")  #FOR DEBUG
    #print("Vectorise testing done: %f" % duration_test ," seconds")  #FOR DEBUG



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """

    df = read_speeches(path=Path.cwd() / "texts" / "speeches")
    LoadData_and_ExtractFeatures(df)