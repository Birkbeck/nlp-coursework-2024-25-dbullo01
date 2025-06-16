#Re-assessment template 2025
import os
import string
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import spacy
from IPython.display import display
import pickle
import re                                   # for regular expressions
import cmudict
cmu_dict = cmudict.dict()
from collections import Counter, defaultdict


from spacy.symbols import VERB, nsubj,nsubjpass, csubj, csubjpass

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000   #Increased spacy model max length to 3000000 for parse() function. Default for spacy model is 1,000,000



def light_clean_text(text):
    """Returns lightly cleaned text where multiple spaces are removed including '\n'.
        Requires text to clean

        Args:
            text (str): The text to lighty clean.

        Returns:
            str: The text that has been lightly cleaned
        """
    # light clean as need to look at syntax and style in (e) and (f) hence need to preserve linguistic structure -
    # Chose to only remove extra white space and '\n' from text to preserve linguistic structure
    text = ' '.join(text.split())

    return text



# (c) flesch_kincaid: This function should return a dictionary mapping the title of each novel to the Flesch-Kincaid
# reading grade level score of the text. Use thr NLTK library for tokenization and the CMU pronouncing dictionary
# for estimating syllable counts

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    # REF - https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests for Flesch-Kincaid Grade level formula
    # Formula;
    # FK_grade_level_score = 0.39 * (total_no_of_words/total_no_of_sentences) + 11.8 * (total_syllables/total_no_of_words) - 15.59

    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    total_no_of_words = len(words)
    total_no_of_sentences = len(sentences)

    total_no_of_syllables = 0
    #check if each word is in dictionary and get no of syllables for word by calling count_syl() function
    d = cmudict.dict()
    for word in words:
        no_of_syllables = count_syl(word,d)
        total_no_of_syllables = total_no_of_syllables + no_of_syllables

    FK_grade_level_score = 0.39 * (total_no_of_words/total_no_of_sentences) + 11.8 * (total_no_of_syllables/total_no_of_words) -  15.59

    return FK_grade_level_score





def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    #REF https://stackoverflow.com/questions/49581705/using-cmudict-to-count-syllables
    # Check if word (key value) is in dictionary
    word = word.lower()
    no_of_syllables_in_word = 0
    if d.get(word):
        #get the value (syllables) for key (word) found in cmudict
        pronounciations_for_the_word_from_cmddict = d.get(word)

        #get no of syllables in word using first sublist (i.e first pronounciation for a word as there could be more than one pronounciation for a word)
        #for a word returned from cmudict as not counting syllables for each accent for a word)

        #print("pronounciation(s) for word: ", pronounciations_for_the_word_from_cmddict)  #FOR DEBUG

        #Search each phoneme in pronounciation to find vowel phoneme(s0 (syllables) i.e has a digit as a last character
        #Decided only need to check for vowel phonemes (syllables) in  the first instance of a pronounciation for a workd where two or more may exist

        no_of_syllables_in_word = 0
        vowelphoneme_count = 0
        for phoneme in pronounciations_for_the_word_from_cmddict[0]:
            #checking each phoneme in pronouciation is a vowel phoneme (syllable) i.e phoneme countains a digit
            #as it's last character
            if phoneme[-1].isdigit():
                vowelphoneme_count = vowelphoneme_count + 1
        no_of_syllables_in_word = vowelphoneme_count

        ###print(word) #FOR DEBUG
        ###print("available phonemes for word : %s" % phonemes) # FOR DEBUG
        #print("No of Syllables in word %d" % no_of_syllables_in_word)  #FOR DEBUG
    else:
        #determine no of vowel clusters (for word that is in not in cmu dictionary)
        vowel_clusters=[]
        no_of_syllables_in_word = 0
        # REF https://www.nltk.org/book_1ed/ch03.html     # Section 3.5 Useful Applications of Regular Expressions
        #fd = nltk.FreqDist(re.findall(r'[aeiou]{1,}', word))  #FOR DEBUG - Shows freq counts of each unique vowel cluster
        #print(fd.items())
        # finding all vowel_clusters of size 1 or more (in a word) as a word in English has at least one vowel
        # as any word not in cmudict should always have at least one vowel
        vowel_clusters.append(re.findall(r'[aeiou]{1,}', word))
        no_of_vowel_clusters = sum([len(vowel_cluster) for vowel_cluster in vowel_clusters])
        ###print(word)  #FOR DEBUG
        ###print(vowel_clusters)   # FOR DEBUG
        ###print("no of vowel_clusters:", no_of_vowel_clusters) #FOR DEBUG
        no_of_syllables_in_word = no_of_vowel_clusters

    return no_of_syllables_in_word



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


    for x in data:
        data_clean.append(light_clean_text(x))

    #print(title)  #FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO SEE THE LIST VALUES FOR TITLE
    #print(author) #FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO SEE THE LIST VALUES FOR AUTHOR
    #print(year)   #FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO SEE THE LIST VALUES FOR YEAR

    data = {
       "text": data_clean,
       "title": title,
       "author": author,
       "year": year
    }

    # create dataframe for novels data and sort by year
    df = pd.DataFrame(data).sort_values(by='year', ascending=True, ignore_index=True)
    pd.set_option('display.max_columns', None)  # Display all columns. None - unlimited
    pd.set_option('display.max_rows', None)   # Display all rows. None - unlimited
    pd.set_option('display.width', None)   # Display width in characters for pandas. None - auto-detects width


    return df


  # (e) parse: The goal of this function is to process the texts with spaCy's tokenizer and parser, and store the
    #    processed texts. Your completed function should:
    # i. Use the spaCy nlp method to add a new column to the dataframe that contains parse and tokenized Doc objects for
    #    each text

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    # i. Use the spaCy nlp method to add a new column to the dataframe that contains parse and tokenized Doc objects for
    #    each text [DONE]
    #REF - https: // spacy.io / usage / processing - pipelines  # processing to understand spaCy docs
    #REF - https: // pandas.pydata.org / docs / getting_started / intro_tutorials / 05_add_columns.html
    #REF - https://docs.python.org/3/library/pickle.html#examples   to pickle to a file and load from pickle file
    #REF - https://spacy.io/usage/spacy-101 for info on nlp
    #REF - https://realpython.com/natural-language-processing-spacy-python/#dependency-parsing-using-spacy

    # Need to access text column from dataframe that contains the novels text and create doc object for each text.
    # A doc object in spaCy is created using doc = nlp("This is a text") and nlp is a function
    # Apply nlp function to novel text rows and store doc object for each text in new dataframe column called
    # 'Doc'

    df['parsed'] = df['text'].apply(nlp)
    #print(type(df['doc'][0]))   #TEST - To see if a value in doc column is of spacy Doc type
    #print(df['doc'].values[0])  #TEST - Get first doc column value


    # ii. Serialise the resulting dataframe (i.e. write it out to disk) using the pickle format [DONE]
    store_path = os.path.join(store_path,  out_name)
    with open(store_path +'.pkl', 'wb') as file:
        pickle.dump(df, file)

    # iii. Return the dataframe  [DONE]
    print(df)

    # iv. Load the dataframe from the pickle file and use it for the remainder of this coursework part. [TO ALTER WHEN i is done]
    #     Note: one or more of the texts may exceed the default meximum length for spaCy's model. You will need to either
    #     increase this length of parse the text in sections

    with open(store_path + '.pkl', 'rb') as file:
        # The protocol version used is detected automatically
        df = pickle.load(file)

    #display(df) FOR DEBUG

    return df


#(b) nltk_ttr: This function should return a dictionary mapping the title of each novel to its type-token ratio.
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


def create_cooccurrence_matrix(path=Path.cwd() / "texts" / "novels"):
    text = ""
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
                # create absolute file path for each file found at specifed relative path (path)
                # REF https://stackoverflow.com/questions/17429044/constructing-absolute-path-with-os-path-join
                absolutepath = os.path.abspath(os.path.join(root, name))
                # store absolute paths to files in list (as a list of file aboslute paths)
                filenames.append(absolutepath)

                # Split name of file into parts representing title, author and year using hyphen ('-') to split name on
                items = name.split('-')
                # title part from file name
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
            # Create Cooccurrence Matrix for read text
            cooccurrence_matrix(afile.read())
            data.append(afile.read())  # read the file then add it to the list
            afile.close()  # close the file when you're done
    return

def cooccurrence_matrix(text):
    """Create term term  co-occurence matrices"""
    """

    Args:
        text: input text to create cooccurrence matrix for

    Returns:
        coocurrence:  coocurrence matrix as dictionary
    """
    # Initialize co-occurrence dictionary
    # Lab 6 video af Lab 3 code

    cooccurrence = defaultdict(lambda: defaultdict(int))

    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    for sentence in sentences:
        # Tokenize sentence into words and normalize 9lowercase and remove punctuation)
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word not in string.punctuation]

        unique_words = set(words)

        for word in unique_words:
            for co_word in unique_words:
                if word != co_word:
                    cooccurrence[word][co_word] += 1

    # Display co-occurrence counts
    for word, co_words in cooccurrence.items():
        print(f"{word}: {dict(co_words)}")

    print(cooccurrence)
    return cooccurrence



def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""


    pass


# (f) (ii) The title of each novel and a list of the ten most common syntactic subjects of the verb "to hear" in any tense
# in the text, ordered by their frequency.

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    """
            Args:
                doc:  dataframe colunn containing tokenizsed and parsed spacy doc  
                verb: verb to find common subjects for
            Returns:
                list: List containing common subjects for the specified verb
    """

    # REF-https://stackoverflow.com/questions/66181946/identify-subject-in-sentences-using-spacy-in-advanced-cases
    # REF-https://spacy.io/usage/linguistic-features#dependency-parse  - The example table showed subjects, verbs and
    # children in relation to subject.
    # REF What is Subject, Verb, Object, Complement, Modifier: Grammatical Functions [basic English grammar]
    # https://www.youtube.com/watch?v=vSBATq2KvjQ - Watched to know what a Object and Subject is

    itemList = []

    # counting syntactic subjects
    syntactic_subjects = Counter()

    # Get doc from dataframe column
    for token in doc:
        if token.lemma_ == "hear":  # FOR DEBUG - SHOWS ALL AVAILABLE DEPENDENCIES FOR DIFF TENSES OF VERB "Hear"
        #Show subject dependencies for Verb "hear" in different tenses (using lemma for "hear")
          if token.dep_ in ("nsubj","nsubjpass", "csubj", "csubjpass") and token.pos_ == "VERB" and token.lemma_ == verb:
             #head = token.head.text
            # The Verb is the head and dep_ represents the branch(es) in the dependency diagram from head (verb) to
            # other words in the text
            # Branch could be going to a word that could be a subject. There are 4 types of subject in SpaCy;
            # nsubj - nominal subject, nsubjpass - nominal subject passive, csubj - clausal subject,
            # csubjpass - clausal subject passive
            print(token.head.text, token.dep_, token.pos_, token.text, token.lemma_) #FOR DEBUG
            syntactic_subjects[token.head.text, token.dep_, token.text, token.lemma_] += 1
            itemList.append([syntactic_subjects])

    # printing the 10 most common syntactic subjects for the verb "to hear" in the text
    print("10 MOST COMMON SYNTACTIC SUBJECTS FOR VERB : " + verb)

    return itemList



#(f) Working with parses: the final lines of the code template contains three for loops. Write the functions needed to
#    complete these loops so that they print:
#   (i) The title of each novel and a list of the ten most common syntactic objects overall in the text

def syntactic_objects_counts(doc):
    """Extracts the ten most common syntactic objects in a parsed document. Returns a list of tuples."""
    """
        Args:
            doc:  SpaCy doc (text tokenized and parsed)
        Returns:
            itemList (list): containing "title" of document as string and  list of tuples where each tuple is Syntactic 
            Object and freq count for that syntactic object in the given text.
    """
    #REF What is Subject, Verb, Object, Complement, Modifier: Grammatical Functions [basic English grammar]
    #https://www.youtube.com/watch?v=vSBATq2KvjQ - Watched to know what a Object and Subject is
    # REF https://en.wikipedia.org/wiki/Object_(grammar) to identify types of object in english grammar
    #REF https://spacy.io/models/en#en_core_web_sm  FOR PARSE TAGS FOR SPACY MODEL To identify obj parser tags

    itemList = []

    #counting syntactic objects  in each novel text (parsed data from spacy doc stored in 'parsed' column of dataframe)
    syntactic_objects = Counter()

    #Using zip to get title and doc from dataframe (in each row of title and doc dataframe columns)
    for title, doc in zip(df['title'],df['parsed']):
        for token in doc:
            if token.dep_ == "dobj" or token == "iobj" or token.dep_ == "pobj":
                # count each syntactic object in doc
                syntactic_objects[(token.lemma_, token.dep_)] += 1
        #Get the ten most common syntactic objects for each title and store both the title and list of syntactic objects
        itemList.append([title, syntactic_objects.most_common(10)])

    return itemList



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
    print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle.pkl")
    print(syntactic_objects_counts(df))

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    create_cooccurrence_matrix(path=Path.cwd() / "texts" / "novels")
    """
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """




