import json
import os
import src.data.data_loader as dl
import pandas as pd
import matplotlib.pyplot as plt
from src.definitions import DATA_DIR
import numpy as np

def _source_relevance_statistics(lang, type):

    if lang == 'german':
        sources = dl.german_types[type]
    else:
        sources = dl.english_types[type]
    path = os.path.join(DATA_DIR, 'raw', lang)
    results = []

    for source in sources:
        result_dict = {}
        total_articles = 0
        relevant_articles = 0
        total_comments = 0
        relevant_comments = 0
        root_comments = 0
        relevant_articles_w_cmnt = 0
        with open(os.path.join(path, source + '.json')) as f:
            data = json.load(f)

            for article in data:
                total_articles += 1
                total_comments += len(article['comments'])
                if article['relevant'] == 1:
                    relevant_articles += 1
                    relevant_comments += len(article['comments'])

                    if not article['comments'] == []:
                        relevant_articles_w_cmnt += 1

                    for comment in article['comments']:
                        if 'comment_replyTo' not in comment:
                            root_comments += 1

        result_dict['Source'] = source
        result_dict['Total articles'] = total_articles
        result_dict['Relevant articles'] = relevant_articles
        result_dict['% rel. articles'] = round(relevant_articles / total_articles, 4) * 100
        result_dict['Total comments'] = total_comments
        result_dict['Relevant comments'] = relevant_comments
        if total_comments > 0:
            result_dict['% rel. cmnt.'] = round(relevant_comments / total_comments, 4) * 100
        else:
            result_dict['% rel. cmnt.'] = 0
        result_dict['Avg. # cmnt.'] = int(relevant_comments/ relevant_articles)
        result_dict['Root cmnt.'] = root_comments
        if relevant_comments > 0:
            result_dict['% root cmnt.'] = round(root_comments / relevant_comments, 4) * 100
        else:
            result_dict['% root cmnt.'] = 0
        result_dict['Rel. art. w/ cmnt.'] = relevant_articles_w_cmnt
        result_dict['% rel. art. w/ cmnt.'] = round(relevant_articles_w_cmnt / relevant_articles, 4) * 100
        results.append(result_dict)
    df = pd.DataFrame(results)
    return df


def _source_average_length_statistics(lang, type):
    if lang == 'german':
        sources = dl.german_types[type]
    else:
        sources = dl.english_types[type]
    results = []
    for source in sources:
        result_dict = {}
        article_data = dl.get_articles_by_sources([source])
        article_texts = article_data['article_texts']
        num_articles = len(article_texts)
        article_text_lengths = [len(t.split()) for t in article_texts]
        sum_article_text_lengths = np.sum(article_text_lengths)
        avg_article_length = int(sum_article_text_lengths / num_articles)

        comment_data = dl.get_comments_by_sources([source], aggregate='comments')
        comment_texts = comment_data['comment_texts']
        num_comments = len(comment_texts)
        if num_comments > 0:
            comment_text_lengths = [len(t.split()) for t in comment_texts]
            sum_comment_text_lenghts = np.sum(comment_text_lengths)
            avg_comment_length = int(sum_comment_text_lenghts/ num_comments)
        else:
            avg_comment_length = 0

        result_dict['Source'] = source
        result_dict['Avg. article length'] = avg_article_length
        result_dict['Avg. cmnt. length'] = avg_comment_length
        results.append(result_dict)
    df = pd.DataFrame(results)
    return df

def source_statistics(lang, type):
    """
    Return a tuple of Pandas dataframes with descriptive statistics for the given language and source type. The first dataframe contains article statistics while the second contains comment statistics.

    Article dataframe columns:  'Source', 'Total articles', 'Relevant articles', '% rel. articles', 'Avg. article length', 'Rel. art. w/ cmnt.', '% rel. art. w/ cmnt.'

    Comment dataframe columns:  'Total comments', 'Relevant comments', '% rel. cmnt.', 'Root cmnt.', '% root cmnt.',
               'Avg. # cmnt.', 'Avg. cmnt. length'

    :param lang: 'german' or 'english'
    :param type: 'editorial', 'blog', or 'forum'
    :return: tuple(pd.DataFrame, pd.DataFrame)
    """
    relevance_df = _source_relevance_statistics(lang, type)
    length_df = _source_average_length_statistics(lang, type)
    joined_df = relevance_df.join(length_df.set_index('Source'), on='Source')
    columns = ['Source', 'Total articles', 'Relevant articles', '% rel. articles', 'Avg. article length',
               'Rel. art. w/ cmnt.', '% rel. art. w/ cmnt.',
               'Total comments', 'Relevant comments', '% rel. cmnt.', 'Root cmnt.', '% root cmnt.',
               'Avg. # cmnt.', 'Avg. cmnt. length']

    joined_df = joined_df.reindex(columns, axis=1)
    article_df = joined_df.iloc[:, 0:7].copy()
    comment_df = joined_df.iloc[:, 7:].copy()
    source_column = joined_df.loc[:, ['Source']].copy()
    comment_df.insert(loc=0, column='Source', value=source_column)
    return article_df, comment_df

def plot_article_type_distribution():
    languages = ['german', 'english']
    types_per_language = []
    for lang in languages:
        types = ['editorial', 'forum']
        type_counts = []

        for idt, typex in enumerate(types):
                data = dl.get_articles_by_type(lang, typex)
                texts = data['article_texts']
                type_counts.append(len(texts))
        types_per_language.append(type_counts)

    editorial_counts = types_per_language[0][0] + types_per_language[1][0]
    forum_counts = types_per_language[0][1] + types_per_language[1][1]
    x = np.arange(2)
    width = 0.2
    plt.bar(x, editorial_counts, width, label='Editorial', color='r')
    plt.bar(x+1*width, forum_counts, width, label='Forums', color='g')
    plt.ylabel('Number of Articles')
    plt.title('Number of Articles per Type')
    plt.legend()
    plt.xticks(x+width, ('German', 'English'))
    ax = plt.axes()
    ax.yaxis.grid(True)
    plt.show()
    #plt.savefig('article_type_distribution.pdf', bbox_inches='tight')


def plot_comment_type_distribution():
    languages = ['german', 'english']
    types_per_language = []
    for lang in languages:
        types = ['editorial', 'forum']
        type_counts = []

        for idt, typex in enumerate(types):
                data = dl.get_comments_by_type(lang, typex, aggregate='article')
                texts = data['comment_texts']
                type_counts.append(len(texts))
        types_per_language.append(type_counts)

    editorial_counts = types_per_language[0][0] + types_per_language[1][0]
    forum_counts = types_per_language[0][1] + types_per_language[1][1]
    x = np.arange(2)
    width = 0.2
    plt.bar(x, editorial_counts, width, label='Editorial', color='r')
    plt.bar(x+2*width, forum_counts, width, label='Forums', color='g')
    plt.ylabel('Number of Comments')
    plt.title('Number of Comments per Type')
    plt.legend()
    plt.xticks(x+width, ('German', 'English'))
    ax = plt.axes()
    ax.yaxis.grid(True)
    plt.show()
    #plt.savefig('comment_type_distribution.pdf', bbox_inches='tight')


def plot_type_time_distribution(lang):
    """
    Plot the counts of documents for each type over the time for a given language.
    :param lang: 'english' or 'german'
    """
    types = ['editorial', 'forum']
    type_counts = {'editorial': 0, 'forum': 0}
    for typex in types:
        data = dl.get_articles_by_type(lang, typex, metadata=['article_time'])
        year_documents = dl.map_documents_to_year(data['article_time'])
        year_counts = []
        for year in year_documents:
            # print(type, len(year_documents[year]))
            year_counts.append(len(year_documents[year]))
        type_counts[typex] = year_counts

    years = np.arange(2007, 2018)
    plt.plot(years, type_counts['editorial'], 'rs-', label='Editorial')
    #plt.plot(years, type_counts['blog'], 'bv-', label='Blog')
    plt.plot(years, type_counts['forum'], 'go-', label='Forum')

    plt.ylabel('Number of Articles')
    plt.title('Number of {} Articles per Year and Type'.format(lang.title()))
    plt.legend()
    ax = plt.axes()
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('{}_time_distribution.pdf'.format(lang), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    plot_type_time_distribution("english")
    plot_type_time_distribution("german")
