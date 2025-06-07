

import pandas as pd
from IPython.display import display
import numpy as np
from pathlib import Path


# Part Two - Feature Extraction and Classification

# In the second part of the coursework, your task is to train and test machine learning classifiers on a dataset of
# political speeches. The objective is to learn to predict the political party from the text of the speech. The texts you
# need for this part are in the speeches sub-directory of the texts directory of the coursework. Moodle template.
# For this part, you can structure your python functions in any way that you like, but pay attention to exactly what
# information (if any) you are asked to print out in each part. Your final scripts should print out the answers to each
# part where required, and nothing else


def read_speeches(path=Path.cwd() / "texts" / "novels"):

    # (a) Read the handsard40000.csv dataset in the texts directory into a dataframe. Subset and rename the dataframe as follows;
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

    # (ii) remove any rows where the value of the 'party' column is not one of the four most common party names and remove
    #       the 'Speaker' value. [DONE]

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


    # Return the dimensions of the resulting dataframe using the shape method
    #REF https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
    return df.shape

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # Print the dimensions of the resulting dataframe using the shape method [DONE]
    dimensions = read_speeches(path=Path.cwd() / "texts" / "speeches")
    print(str(dimensions))