#Re-assessment template 2025
import os
import string
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


# (c) flesch_kincaid: This function should return a dictionary mapping the title of each novel to its type-token ratio.
# Tokenize the text string using the NLTK library only. Do not include punctuation as tokens, and ignore case when
# counting types

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass




#(a) read_novels: Each file in the novels directory contains the text of a novel,and the name of the file is the
#    title, author, and year of publication of the novel, separated by hyphens. Complete the python function read_texts
#    to do the following:
# i.  create a pandas dataframe with the following columns: text, title, author, year [DONE]
# ii. sort the dataframe by the year column before returning it, resetting or ignoring the dataframe index   [DONE]

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year

    Args:
        path (str): path to novels data.

    Returns:
        dataframe: sorted by the year column before returning it, resetting or ignoring the dataframe index

    """
    # REF - Lab 2 code (amended to handle relative directory path and convert to absolute file paths)
    # REF - https://builtin.com/data-science/data-frame-sorting-pandas  for knowing how ignore_index works with sorted dataframe
    relpath = path
    file_type = ".txt"  # if your data is not in a plain text format, you can change this
    filenames = []
    title = []
    author = []
    year = []
    data = []

    # this for loop will run through folders and subfolders looking for a specific file type
    for root, dirs, files in os.walk(relpath, topdown=False):
        # look through all the files in the given directory
        for name in files:
            if name.endswith(file_type):
                #create absolute file path for each file found at specifed relative path (path)
                #REF https://stackoverflow.com/questions/17429044/constructing-absolute-path-with-os-path-join
                absolutepath = os.path.abspath(os.path.join(root, name))
                #store absolute paths to files in list (as a list of file aboslute paths)
                filenames.append(absolutepath)

                #Split name of file into parts representing title, author and year using hyphen ('-') to split name on
                items = name.split('-')
                #title part from file name
                title.append(items[0])
                # store author part of file name to
                author.append(items[1])
                # store year part for file name to years (list) e.g (stripping '.txt' off the end of items[2] value)
                year.append(items[2][:-4])

    # this for loop then goes through the list of files (absolute file paths), reads the files, and then adds the text
    # to Data (list)
    data_clean = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as afile:
            print(filename)
            # clean text from folder
            data.append(afile.read())  # read the file then add it to the list
            afile.close()  # close the file when you're done


    #print(title)  #FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO SEE THE LIST VALUES FOR TITLE
    #print(author) #FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO SEE THE LIST VALUES FOR AUTHOR
    #print(year)   #FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO SEE THE LIST VALUES FOR YEAR

    data = {
       "text": data,
       "title": title,
       "author": author,
       "year": year
    }

    # create dataframe for novels data and sort by year
    df = pd.DataFrame(data).sort_values('year')
    pd.set_option('display.max_columns', None)  # Display all columns. None - unlimited
    pd.set_option('display.max_rows', None)   # Display all rows. None - unlimited
    pd.set_option('display.width', None)   # Display width in characters for pandas. None - auto-detects width

    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


#(b) nltk_trr: This function should return a dictionary mapping the title of each novel to its type-token ratio.
# Tokenize the text using the NLTK library only. Do not include punctuation as tokens, and ignore case when counting
# types


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.
    Args:
            text (str): text string to calculate ttr

    Returns:
            ttr (float): type token ratio,  where ttr = no of unique words / no of words

    """
    #REF Lab2 solution code for removing punctuation
    #REF https://docs.python.org/3/library/stdtypes.html#string-methods for how isalpha works in python

    #nltk.download('punkt')  # PLEASE UNCOMMENT AND RUN TO INSTALL IF PUNKT IS NOT INSTALLED

    tokens = word_tokenize(text.lower())  # lower case text to ignore case (when counting later)  and tokenize
    tokens = [token for token in tokens if token.isalpha()]  # remove punctuation, returns list of tokens

    unique_tokens = set(tokens)  # set of unique tokens (types)

    # print("tokens", tokens)  # FOR DEBUG - UNCOMMENT IF YOU WOULD LIKE TO SEE TOKENS
    # print("unique tokens", unique_tokens)  # FOR DEBUG - UNCOMMENT IF YOU WOULD LIKE TO SEE UNIQUE TOKENS

    if len(tokens) == 0:
        ttr = 0  # stop divide by zero error (when there is no text)
    else:
        #ttr count of unique tokens divided by count of tokens
        ttr = len(unique_tokens) / len(tokens)
    return ttr



def get_ttrs(df):
    """helper function to add ttr to a dataframe
    Args:
            df (pandas dataframe):

    Returns:
            results (dictionary): containing "title" of document and  TTR is calculated on document text ("text"), both
            are  obtained from a dataframe (df) and stored as key value pairs for each title in a dictionary (results)
    """

    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    #nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

