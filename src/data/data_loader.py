'''
This implementation is copied from Christians project
'''
import json
import os
from collections import defaultdict
from datetime import datetime
from src.definitions import DATA_DIR

german_types = {
    'editorial': ['spiegel', 'zeit', 'welt', 'taz', 'tagesspiegel', 'handelsblatt', 'freitag', 'tagesschau', 'br',
                  'wdr', 'swr', 'ndr', 'derstandard', 'diepresse', 'kurier', 'nachrichtenat', 'salzburgcom', 'krone',
                  'tagesanzeiger', 'nzz', 'aargauer', 'luzernzeitung', 'srf', 'forum_ernaehrung',
                  'heise', 'eatsmarter', 'huffingtonpost_de', 'waz', 'merkur', 'rp', 'focus', 'campact'],

    'blog': ['individualisten', 'berlinbio', 'biologisch-lecker', 'scilogs', 'lebeheute',
              'greenpeace', 'karmakonsum', 'nachhaltigleben', 'utopia', 'drfeil'],

    'forum': ['reddit_de', 'gutefrage', 'werweisswas',  'glamour', 'webkoch', 'chefkoch', 'paradisi',
              'kleiderkreisel', 'biooekoforum', 'bfriendsBrigitte', 'schule-und-familie']}

english_types = {'editorial': ['usatoday', 'nytimes', 'nypost', 'washingtonpost', 'latimes', 'chicagotribune',
                               'huffingtonpost', 'organicauthority'],
                 'blog': ['foodbabe', 'organicconsumers', 'foodrevolution', 'disqus'],
                 'forum': ['reddit', 'usmessageboard', 'cafemom', 'quora', 'fb']}


german_sources = [e for v in german_types.values() for e in v]
english_sources = [e for v in english_types.values() for e in v]


def _map_source_to_language(source):
    if source in german_sources:
        return 'german'
    elif source in english_sources:
        return 'english'
    else:
        raise Exception('Source {} not included'.format(source))


def get_articles_by_type(language, source_type, metadata=None, merge_tokens=True, kind="with_2bigrams"):
    """
    Return all article texts for editorial and blog sources. For forum threads use :py:func:`get_forum_threads_by_language`.

    See :py:func:`get_articles_by_sources` for the explanation of the parameters and the return value.

    """

    if language == 'german':
        return get_articles_by_sources(german_types[source_type], metadata, kind, merge_tokens)
    elif language == 'english':
        return get_articles_by_sources(english_types[source_type], metadata, kind, merge_tokens,)
    else:
        raise Exception('Language {} not supported.'.format(language))


def get_articles_by_sources(sources, metadata=None, kind="with_2bigrams", merge_tokens=True):
    """
    Return all article texts of the specified sources and the corresponding metadata. The documents are returned in a dictionary.
    The key 'article_texts' contains the text in the form specified by merge_paragraphs and 'source2articles' contains
    a mapping of each source to the article ids (position in list).

    .. Note:: To load forum data use :py:func:`get_forum_threads_by_sources`.

    :param sources: List of sources to load
    :type sources: list
    :param metadata: List of meta information to load together with article texts. The possible values correspond to the fields of the json files
    :type metadata: list
    :param kind: Specifies the kind of processing applied on the data. The possible values 'tokenize_lemmatize',
     'stopwords', and 'processed' correspond to the folders of the data directory.
     :type kind: str
     :param merge_tokens: If true the tokens are joined to string otherwise the tokens for each document are contained as a list.
     True: ["t1 t2", "t3 t4"] False: [["t1", "t2"],["t3", "t4"]]. Use True if the documents are passed to the vectorizer and use false if they are passed
     to the topic labeler
     :type merge_tokens: bool
    :returns: Dict {article_texts: [...], metadata_0: [...], metadata_1: [...], ...,
     source2articles: {'source0': [], 'source1': [] }}

    """
    source2articles = {}
    idx = 0
    results = defaultdict(list)
    for source in sources:
        ids = []
        lang = _map_source_to_language(source)
        path = os.path.join(DATA_DIR, kind, lang, source + '.json')  # possibly change to only have one german file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for resource in data:
                # Get text
                article_text_tokenized = resource['article_text_tokenized']
                if merge_tokens:
                    article_text_tokenized_joined = " ".join(article_text_tokenized)
                    results['article_texts'].append(article_text_tokenized_joined)
                else:
                    results['article_texts'].append(article_text_tokenized)

                if kind == "tagged":
                    results['article_pos'].append(resource['article_text_pos'])
                # Get the meta information
                if metadata:
                    for idx, meta in enumerate(metadata):
                        if meta in resource:
                            results[meta].append(resource[meta])
                        else:
                            results[meta].append("NO {} INFORMATION FOR ARTICLE".format(meta))

                # Add id to ids
                ids.append(idx)
                idx += 1
        source2articles[source] = ids
    if kind == "tagged":
        results['article_pos'] = list2tupel(results["article_pos"])

    results['source2articles'] = source2articles
    return results


def get_comments_by_type(language, source_type, aggregate='article', kind='with_2bigrams', merge_tokens=True):
    """
    Return all comment texts for editorial articles or blog posts. For forum threads use :py:func:`get_forum_threads_by_language`.

    See :py:func:`get_comments_by_sources` for the explanation of the parameters and the return value.

    """

    if language == 'german':
        return get_comments_by_sources(german_types[source_type], aggregate,  kind,  merge_tokens=merge_tokens)
    elif language == 'english':
        return get_comments_by_sources(english_types[source_type], aggregate,  kind,  merge_tokens=merge_tokens)
    else:
        raise Exception('Language {} not supported.'.format(language))


def get_comments_by_sources(sources, aggregate='article', kind="with_2bigrams", merge_tokens=True):
    """
    Get all comments for all articles of the specified sources. The documents are returned in a dictionary.
    The key 'comment_texts' contains the text in  a list and 'comment2articles' contains a mapping of each comment document to the article ids if :py:func:`get_articles_by_sources` is called with the  same sources in the same order.

    - Possibilities to aggregate comments:
        - article : For each article all comments (root+children) will be concatenated to one document.
        - article_roots  : For each article only root comments will be concatenated to a single document.
        - only_root: Each comment is a single document and only root comments are returned.
        - comments: Every comment is a single document.

    :param sources: List of sources to load
    :type sources: list
    :param aggregate: One of 'article', 'article_roots', 'only_root', and 'comments'
    :type aggregate: str
     :param kind: Specifies the kind of processing applied on the data. The possible values 'tokenize_lemmatize', 'stopwords', and 'processed' correspond to the folders of the data directory.
     :type kind: str
     :returns: Dict{comment_texts: [...], comment2article: [...]}

    """
    article_id = 0
    documents = []
    article_ids = []
    comment_pos = []
    for source in sources:
        lang = _map_source_to_language(source)
        path = os.path.join(DATA_DIR, kind, lang, source + '.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for resource in data:
                if resource['comments'] == []:
                    article_id += 1
                    continue

                if aggregate == 'article':
                    article_comments = []
                    article_comments_pos = []
                    for comment in resource['comments']:
                        if kind == "tagged":
                            com, com_pos = _concatenate_comment(comment,  False, "tagged")
                            article_comments = article_comments + com
                            article_comments_pos = article_comments_pos + com_pos
                        else:
                            article_comments = article_comments + _concatenate_comment(comment, merge_tokens=False)
                    if merge_tokens:
                        documents.append(" ".join(article_comments))
                    else:
                        documents.append(article_comments)
                    if kind == "tagged":
                        comment_pos.append(article_comments_pos)
                    article_ids.append(article_id)

                elif aggregate == 'article_roots':
                    article_comments = []
                    article_comments_pos = []
                    for comment in resource['comments']:
                        if 'comment_replyTo' not in comment:
                            if kind == "tagged":
                                com, com_pos = _concatenate_comment(comment, False, "tagged")
                                article_comments = article_comments + com
                                article_comments_pos = article_comments_pos + com_pos
                            else:
                                article_comments = article_comments + _concatenate_comment(comment, merge_tokens=False)
                    if merge_tokens:
                        documents.append(" ".join(article_comments))
                    else:
                        documents.append(" ".join(article_comments))

                    if kind == "tagged":
                        comment_pos.append(article_comments_pos)

                    article_ids.append(article_id)

                elif aggregate == 'only_root':
                    for comment in resource['comments']:
                        if 'comment_replyTo' not in comment:
                            if kind == "tagged":
                                con_comment, con_pos = _concatenate_comment(comment, merge_tokens=False, kind="tagged")
                                comment_pos.append(con_pos)
                            else:
                                con_comment = _concatenate_comment(comment, merge_tokens=merge_tokens)
                            documents.append(con_comment)
                            article_ids.append(article_id)

                elif aggregate == 'comments':
                    for comment in resource['comments']:
                        if kind == "tagged":
                            con_comment, con_pos = _concatenate_comment(comment, merge_tokens=False, kind="tagged")
                            comment_pos.append(con_pos)
                        else:
                            con_comment = _concatenate_comment(comment, merge_tokens=merge_tokens)
                        documents.append(con_comment)
                        article_ids.append(article_id)
                else:
                    raise Exception('Aggregation level {} not supported'.format(aggregate))
                article_id += 1
    if kind == "tagged":
        comment_pos = list2tupel(comment_pos)
        return {'comment_texts': documents, 'comment_pos': comment_pos, 'comment2article': article_ids}
    else:
        return {'comment_texts': documents, 'comment2article': article_ids}


def _concatenate_comment(comment, merge_tokens=True, kind="with_2bigrams"):
    comment_text = comment['comment_text_tokenized']
    if 'comment_title' in comment:
        comment_text = comment_text + comment['comment_title_tokenized']

    if merge_tokens:
        comment_text = " ".join(comment_text)
    # if kind is tagged also return pos tags
    if kind == "tagged":
        comment_pos = comment['comment_text_pos']
        if 'comment_title' in comment:
            comment_pos = comment_pos + comment['comment_title_pos']
        return comment_text, comment_pos

    return comment_text


def get_forum_threads_by_language(language,  metadata=None, kind='with_2bigrams', merge_tokens=True):
    """
    Return all forum posts of the specified language.

    See :py:func:`get_forum_threads_by_sources` for the explanation of the parameters and the return value.

    """
    if language == 'german':
        return get_forum_threads_by_sources(german_types['forum'], metadata, kind, merge_tokens=merge_tokens)
    elif language == 'english':
        return get_forum_threads_by_sources(english_types['forum'], metadata, kind,  merge_tokens=merge_tokens)
    else:
        raise Exception('Language {} not supported.'.format(language))


def get_forum_threads_by_sources(sources,  metadata=None, kind='with_2bigrams', merge_tokens=True):
    """
    Return all threads of the specified sources and the corresponding metadata. Each document combines the article_title,
    article_text, and comments.  The documents are returned in a dictionary.
    The key 'thread_texts' contains the text in the form specified by merge_paragraphs and 'source2threads' contains
    a mapping of each source to the thread ids (position in list).

    :param sources: List of sources to load
    :type sources: list
    :param metadata: List of meta information to load together with the threads. The possible values correspond to the fields of the json files.
    :type metadata: list
    :param kind: Specifies the kind of processing applied on the data. The possible values 'tokenize_lemmatize', 'stopwords', and 'processed' correspond to the folders of the data directory.
     :type kind: str
    :returns: Dict {thread_texts: [...], metadata_0: [...], metadata_1: [...], ...,
     source2thread: {'source0': [], 'source1': [] }}

    """
    source2articles = {}
    id = 0
    results = defaultdict(list)
    for source in sources:
        ids = []
        lang = _map_source_to_language(source)
        path = os.path.join(DATA_DIR, kind, lang, source + '.json')  # possibly change to only have one german file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for resource in data:
                thread_intro = resource['article_title_tokenized'] + resource['article_text_tokenized']

                if kind == "tagged":
                    thread_pos = resource['article_title_pos'] + resource['article_text_pos']

                for comment in resource['comments']:
                    if kind == "tagged":
                        con_comment, comment_pos = _concatenate_comment(comment, merge_tokens=False, kind="tagged")
                        thread_pos = thread_pos + comment_pos
                    else:
                        con_comment = _concatenate_comment(comment, merge_tokens=False)
                    thread_intro = thread_intro + con_comment

                if merge_tokens:
                    results['thread_texts'].append(" ".join(thread_intro))
                else:
                    results['thread_texts'].append(thread_intro)

                if kind == "tagged":
                    results['thread_pos'].append(thread_pos)

                # Get the meta information
                if metadata:
                    for idx, meta in enumerate(metadata):
                        if meta in resource:
                            results[meta].append(resource[meta])
                        else:
                            results[meta].append("NO {} INFORMATION FOR ARTICLE".format(meta))

                # Add id to ids
                ids.append(id)
                id += 1
        source2articles[source] = ids
    if kind == "tagged":
        results['thread_pos'] = list2tupel(results['thread_pos'])
    results['source2threads'] = source2articles
    return results

def list2tupel(listA):
    result_list = []
    for l in listA:
        interims_list = []
        for x in l:
            interims_list.append(tuple(x))
        result_list.append(interims_list)
    return result_list

def map_documents_to_year(article_times, counts=False):
    """
    Return a mapping of years to article_ids or the number of articles for a year.
    Given a list of article dates, as retrieved by calling :py:func:`get_articles_by_sources` with option metadata = ['article_time'], the function returnes a mapping of {year1: [articleId1, articleId2 ,...], year2: [ ...]}.


    :param article_times: A list of article_times
    :type article_times: list
    :param counts: If true the number of documents per year is returned instead of lists of article ids.
    :type counts: bool
    :return: Dict{year1: [articleId1, articleId2 ,...], year2: [ ...]} or Dict{year1: int, year2: int, ...}

    """
    years = range(2007, 2018)
    result = {year: [] for year in years}
    for docid, date in enumerate(article_times):
        if date == '':
            continue
        dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        result[dt.year].append(docid)

    if counts:
        result = {year: len(result[year]) for year in result}
    return result


if __name__ == "__main__":
    data = get_forum_threads_by_language('german', kind='tagged', merge_tokens=False)
    texts = data["thread_texts"]
    print(len(texts))
    print(texts[0])


