
import pandas as pd
from IPython.display import display
import numpy as np
from pathlib import Path
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import density
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

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

def LoadData(df):
    """Load data

          Args:
             df: pandas dataframe containing data to create x (data) and y (labels)
          Returns:
             X_train: training data
             X_test:  testing data
             y_train:  labels training data
             y_test:  labels testing data
             features_names:
      """

    # Checking the columns exist in the dataframe
    # display(df.columns) #FOR DEBUG

    # Split the dataframe data into the required  x (data) and y (target labels)
    X = df['speech']
    y = df['party']

    # display(X)  # FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO iNSPECT DATA ('speech)
    # display(y)  # FOR DEBUG - PLEASE UNCOMMENT IF YOU WOULD LIKE TO iNSPECT LABELS DATA ('party')

    # Split the X (speeches) and y (party labels) data into training set 75% and 25% for testing set
    # using stratifIed sampling, max_features set to 3000,
    # Where: max features means, if not None, is used to build a vocabulary that considers the top max_features
    # ordered by term frequency across the corpus. Otherwise, all features are used (taken from
    # TfidVectorizer scikit learn online help pages)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=26, stratify=y)
    return X_train, X_test, y_train, y_test


def ExtractFeatures(X_train, X_test, y_train, y_test):
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
           X_train:
           X_test:
           y_train:
           y_test:
        Returns:
           X_train_extracted_features: Extracted tf-idf features from training data
           X_test_extracted_features: Extracted tf-idf features from testing data
           y_train:  labels training data
           y_test:  labels testing data
           features_names:
    """

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
    print("train time %f " % duration_train)

    # Extracting features from the test data using the same vectorizer
    t0 = time()
    X_test_extracted_features = vectorizer.transform(X_test)
    duration_test = time() - t0
    print("test time %f " % duration_test)

    feature_names = vectorizer.get_feature_names_out()

    print(f"{len(X_train)} documents (training set)")   # FOR DEBUG
    print(f"{len(X_test)} documents (testing set)")     # FOR DEBUG
    print("Vectorise training done: %f" % duration_train ," seconds")  #FOR DEBUG
    print("X_train n_samples: ", pd.DataFrame(X_train).shape[0], "X_train n_features:", pd.DataFrame(X_train).shape[1])  #FOR DEBUG
    print("Vectorise testing done: %f" % duration_test ," seconds")  #FOR DEBUG
    print("X_test n_samples: ", pd.DataFrame(X_test).shape[0], "X_test n_features:", pd.DataFrame(X_test).shape[1])  # FOR DEBUG

    return X_train_extracted_features, X_test_extracted_features, y_train, y_test, feature_names

def ExtractFeatures_bi_grams(df):
        # (d) Adjust the parameters of the TfidVectorizer so that uni-grams, bi-grams and tri-grams will be considered
        #     as features, limiting the total number of features to 3000. Print the classification report as in 2(c) again
        #     using these parameters [DONE]
        # REF https://code.likeagirl.io/good-train-test-split-an-approach-to-better-accuracy-91427584b614 for the types of
        # train test splits
        # REF https://builtin.com/data-science/train-test-split#:~:text=Stratified%20Splitting&text=This%20creates%20training%20and%20testing,categories%20aren't%20represented%20equally.
        # for what stratified sampling means
        # REF https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        # for parameters and their defaults and what max_features and stratified means

        """Load data

            Args:
                X_train:
                X_test:
                y_train:
                y_test:
            Returns:
                X_train_extracted_features: Extracted tf-idf features from training data
                X_test_extracted_features: Extracted tf-idf features from testing data
                y_train:  labels training data
                y_test:  labels testing data
                features:
        """

        # REF https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html for examples
        # of extracting features for train and test data using TfIdfVectorizer

        # Extracting features using a sparse vectorizer TfIdf
        # Using bi-grams instead of uni-grams or tri-grams as bi-grams performed best during hyperparameter tuning
        t0 = time()
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        X_train_extracted_features = vectorizer.fit_transform(X_train)
        duration_train = time() - t0
        print("train time %f " % duration_train)

        # Extracting features from the test data using the same vectorizer
        t0 = time()
        X_test_extracted_features = vectorizer.transform(X_test)
        duration_test = time() - t0
        print("test time %f " % duration_test)

        feature_names = vectorizer.get_feature_names_out()

        print(f"{len(X_train)} documents (training set)")  # FOR DEBUG
        print(f"{len(X_test)} documents (testing set)")  # FOR DEBUG
        print("Vectorise training done: %f" % duration_train, " seconds")  # FOR DEBUG
        print("X_train n_samples: ", pd.DataFrame(X_train).shape[0], "X_train n_features:",
              pd.DataFrame(X_train).shape[1])  # FOR DEBUG
        print("Vectorise testing done: %f" % duration_test, " seconds")  # FOR DEBUG
        print("X_test n_samples: ", pd.DataFrame(X_test).shape[0], "X_test n_features:",
              pd.DataFrame(X_test).shape[1])  # FOR DEBUG

        return X_train_extracted_features, X_test_extracted_features, y_train, y_test, feature_names


# (c) Train RandomForest (with n_estimators=300) and SVM with linear kernel classifiers on the training set,
    # and print the scikit-learn macro-average f1 score and classification report for each classifier on the test set.
    # The label that you are trying to predict is the 'party' value
    # REF https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html tutorial Week 5 lab
    # REF https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/

def benchmark_classification_models(classifier, classifier_name, X_train, y_train, X_test, y_test):
    """benchmark classification model(s)

            Args:
                classifier:
                X_train:
                y_train:
                X_test:
                y_test:

            Returns:
                class_report:
                training_duration_time:
                testing_duration_time:
    """

    print("y_train n_samples :", pd.DataFrame(y_train).shape[0])  #FOR DEBUG
    print("Y_train n_features:", pd.DataFrame(y_train).shape[1])  #FOR DEBUG


    print("Training ", classifier)
    t0 = time()
    classifier.fit(X_train,y_train)
    training_duration_time = time() - t0

    t0 = time()
    predictions = classifier.predict(X_test)
    testing_duration_time = time() - t0


    class_report = classification_report(y_test,predictions, zero_division=0)
    macro_average_f1_score = f1_score(y_test, predictions,average="macro", zero_division=0)
    print("classification report:")
    print(class_report)
    print("macro average f1 score:", macro_average_f1_score)


    #print("pred y:",predictions)         #FOR DEBUG
    #print("y_test (actuals)", y_test)    #FOR DEBUG


    if hasattr(classifier, "coef_"):
        print(f"dimensionality: ", {classifier.coef_.shape[1]})
        print(f"density: ",  {density(classifier.coef_)})

    if classifier_name:
        classifier_descr = str(classifier_name)
    else:
        classifier_descr = classifier.__class__.__name__

    return classifier_descr, macro_average_f1_score, training_duration_time, testing_duration_time


def classifier_pipeline(X_train, y_train, X_test, y_test):
    """benchmark classification model(s)

                Args:
                    X_train:
                    y_train:
                    X_test:
                    y_test:

                Returns:
                    class_report:
                    training_duration_time:
                    testing_duration_time:
        """
    #REF https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #https://stackoverflow.com/questions/31677218/scikit-f-score-metric-error?rq=3
    #https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi?rq=1

    classification_results = []
    for classifier, classifier_name in (
        (RandomForestClassifier(n_estimators=300,class_weight="balanced"),"Random Forest"),
        (LinearSVC(C=0.1, dual=False, max_iter=10000,class_weight="balanced"), "Linear SVC"),
    ):
        print("=" * 80)
        print(classifier_name)
        classification_results.append(benchmark_classification_models(classifier,classifier_name,X_train, y_train, X_test, y_test))


def pipeline_for_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """

    Args:
        X_train:
        X_test:
        y_train:
        y_test:

    Returns:

    """
    #Classification of text documents using sparse features
    #REF - https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
    #Sample pipeline for text feature extraction and evaluation
    #REF - https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html

    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )

    parameter_grid = {
        "vect__ngram_range": ((1,1),(1,2),(1,3)),  # trialing uni-grams, bi-grams and tri-grams for TfidfdVectoriser
        ##"clf__C": (0.01, 0.1, 1, 10, 100),
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=3,
        random_state=0,
        n_jobs=2,
        verbose=1
    )
    print("Performing grid search...")
    print("Hyperparameters to be evaluated:")
    pprint(parameter_grid)

    t0 = time()
    random_search.fit(X_train,y_train)
    duration_of_fitting = time() - t0
    print("duration of fitting: %.3f s" % duration_of_fitting)

    print("Best parameters combination found:")
    best_parameters = random_search.best_estimator_.get_params()
    for param_name in sorted(parameter_grid.keys()):
        print(f"{param_name}:", {best_parameters[param_name]})


    t0 = time()
    predictions = random_search.predict(X_test)
    testing_duration_time = time() - t0


    # Print the classification report and macro_average f1 score
    class_report = classification_report(y_test, predictions, zero_division=0)
    macro_average_f1_score = f1_score(y_test, predictions, average="macro", zero_division=0)
    print("classification report:")
    print(class_report)
    print("macro average f1 score:", macro_average_f1_score)


    #test_accuracy = random_search.score(X_test, y_test)

    #print(
    #    "Accuracy of the best parameters using the inner CV of "
    #    f"the random search: {random_search.best_score_:.3f}"
    #)
    #print(f"Accuracy on test set: {test_accuracy:.3f}")

    return



if __name__ == "__main__":
    """
        uncomment the following lines to run the functions once you have completed them 
    """

    df = read_speeches(path=Path.cwd() / "texts" / "speeches")

    print("")
    # print("Loading Data and Feature Extraction using TfidfVectorizer")
    print("Loading Data")
    print("")
    X_train, X_test, y_train, y_test = LoadData(df)

    print("")
    print("Feature Extraction using TfidfVectorizer")
    print("")
    X_train_extracted_features, X_test_extracted_features, y_train, y_test, feature_names = ExtractFeatures(X_train,
                                                                                                            X_test,
                                                                                                            y_train,
                                                                                                            y_test)
    print("")
    print("Training classification models")
    print("")
    classifier_pipeline(X_train_extracted_features, y_train, X_test_extracted_features, y_test)

    print("")
    print("")
    print("Hyperparameter tuning for the LinearSVC and TfidfVectorizer (vectoriser)")
    print("")
    pipeline_for_hyperparameter_tuning(X_train, X_test, y_train, y_test)

    print("")
    print("")
    print("Feature Extraction using TfidfVectorizer (bi-gram) - after selecting this n-gram from hyperparameter tuning")
    print("")
    X_train_extracted_features2, X_test_extracted_features2, y_train2, y_test2, feature_names2 = ExtractFeatures_bi_grams(df)

    print("")
    print("Training classification models")
    print("")
    classifier_pipeline(X_train_extracted_features2, y_train2, X_test_extracted_features2, y_test2)
