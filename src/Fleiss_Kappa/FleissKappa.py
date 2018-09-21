import pandas as pd
import numpy as np

def set_of_lables(df,column):
    list =[]
    for c in column:
        list.extend((df[c].str.lower()).unique())
    return set(list)

def create_df(index,lablecolum):
    return pd.DataFrame(np.zeros((len(index),len(lablecolum))),columns = lablecolum,index = index)

def fill_matrix(newdf,df):
    index_list = newdf.index
    for i in index_list:
        lable = df.loc[i]
        for l in lable:
            newdf.at[i,l] = newdf.at[i,l] + 1
    return newdf

def calculate_pi(newdf):
    pi = newdf.sum(axis = 0)/(len(newdf.index)*3)
    #for the whole sheet
    sq = pi.apply(lambda x: x**2)
    return sq.sum()

def calculate_Pi(newdef):
    sq = newdf.apply(lambda x: x**2,axis =1)
    ps = ((1/6)*(sq.sum(axis =1)-3)).sum()
    #ocer the whole sheet
    return (1/len(newdf.index))*ps

def kappa(newdf):
    return (calculate_Pi(newdf)-calculate_pi(newdf))/(1-calculate_pi(newdf))


if __name__ == "__main__":
    df = pd.ExcelFile("topic-labeling_expert-labelling.xlsx")
    #, sheet_name="de_edit_art"
    sheets = df.sheet_names

    for s in sheets:
        df = pd.read_excel("topic-labeling_expert-labelling.xlsx",sheet_name = s)
        columns = df[["HD", "PB", "SH"]]
        columns = columns.apply(lambda c: c.str.lower())
        #topics(index)
        indexe = df.index

        lables = set_of_lables(df,columns)
        #print(lables)

        newdf = create_df(indexe,lables)
        #print(newdf)

        filled = fill_matrix(newdf,columns)
        #print(filled)
        #print(filled.loc["Topic 31"])

        cal = calculate_pi(filled)
        calP = calculate_Pi(filled)
        #print(calP)
        print(s,kappa(filled))
