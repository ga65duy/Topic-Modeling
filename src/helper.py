pkl_list = ["topic_models/lda/DEED_lda_german_editorial_articles_190.pkl","topic_models/lda/ENED_lda_english_editorial_articles_130.pkl","topic_models/lda/ENFO_lda_english_forum_110.pkl",
            "topic_models/nmf/DEEDCO_nmf_german_editorial_comments_170.pkl","topic_models/nmf/DEFO_nmf_german_forum_170.pkl","topic_models/nmf/ENEDCO_nmf_english_editorial_comments_170.pkl"]


def ranked_dataframe(df,new_column,ranking,ascending = False):
    dfn = df.copy()
    dfn.insert(len(dfn.columns), 'score', ranking)
    dfn.insert(len(dfn.columns), "alpha", new_column)
    dfn = dfn.sort_values(by=['score'], ascending=ascending)
    dfn = dfn.reindex(columns=(['score'] + list([a for a in dfn.columns if a != 'score'])))
    dfn = dfn.reindex(columns=(['alpha'] + list([a for a in dfn.columns if a != 'alpha'])))
    return dfn
