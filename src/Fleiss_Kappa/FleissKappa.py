import pandas as pd
import numpy as np

"""
Calculating the Interrater reliablity with the help of fleiß kappa according to Wikipedia: 
https://en.wikipedia.org/wiki/Fleiss%27_kappa
"""


def set_of_labels(df, column):
    """
    All lables from the domain expertds in a set

    :param df: dataframe with topics and labels
    :type dataframe
    :param column: selected columns from the labelers
    :type dataframe
    :return: Set of labels that were given from the domain experts
    """
    list = []
    for c in column:
        list.extend((df[c].str.lower()).unique())
    return set(list)


def create_df(index, labelcolum):
    """
    Create a new dataframe with the topic numbers in the rows and the given labels as columns

    :param index: topic numbers
    :type intex
    :param lablecolum: labels from the domain experts
    :type set
    :return: new dataframe
    """
    return pd.DataFrame(np.zeros((len(index), len(labelcolum))), columns=labelcolum, index=index)


def fill_matrix(newdf, df):
    """
    How often the lable was given to a special topic

    :param newdf: dataframe from the function create_df
    :type dataframe
    :param df: dataframe with the topics and labels
    :tyyp dataframe
    :return: Dataframe with the topic numbers as rows and all labels as columns. With tha amount how often a label was selected per topic
    """
    index_list = newdf.index
    for i in index_list:
        lable = df.loc[i]
        for l in lable:
            newdf.at[i, l] = newdf.at[i, l] + 1
    return newdf


def calculate_pi(newdf):
    """
    The value pj is the proportion of all assignments (raters * number of topics) that were made to the jth category.
    :param newdf: dataframe from the function fill matrix
    :type dataframe
    :return: float
    """
    pi = newdf.sum(axis=0) / (len(newdf.index) * 3)
    # for the whole sheet
    sq = pi.apply(lambda x: x ** 2)
    return sq.sum()


def calculate_Pi(newdf):
    """
    :param newdf: dataframe from the function fill matrix
    :return: float
    """
    sq = newdf.apply(lambda x: x ** 2, axis=1)
    ps = ((1 / 6) * (sq.sum(axis=1) - 3)).sum()
    # ocer the whole sheet
    return (1 / len(newdf.index)) * ps


def kappa(newdf):
    """
    Kappa value for the Iterrater reliability. The value is between 0 and 1.
    The higher the value the better is the agreement between the raters.

    :param newdf: dataframe from the function fill matrix
    :return: kappa metric
    """
    return (calculate_Pi(newdf) - calculate_pi(newdf)) / (1 - calculate_pi(newdf))


if __name__ == "__main__":
    df = pd.ExcelFile("topic-labeling_expert-labelling.xlsx")
    # , sheet_name="de_edit_art"
    sheets = df.sheet_names

    for s in sheets:
        df = pd.read_excel("topic-labeling_expert-labelling.xlsx", sheet_name=s)
        columns = df[["HD", "PB", "SH"]]

        columns = columns.apply(lambda c: c.str.lower())

        indexe = df.index

        labels = set_of_labels(df, columns)

        newdf = create_df(indexe, labels)

        filled = fill_matrix(newdf, columns)

        cal = calculate_pi(filled)

        calP = calculate_Pi(filled)

        print(s, kappa(filled))
