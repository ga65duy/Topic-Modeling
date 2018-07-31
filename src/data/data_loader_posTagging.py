import json
import os
import spacy


def process_resource(nlp, data):
    article_per_ressource = []
    for article in data:
        edited = process_article(nlp, article)
        article_per_ressource.append(edited)
    return article_per_ressource

def process_article(nlp, article):
    text_tokenized, text_pos = process_text(nlp, article["article_text"])
    article["article_text_tokenized"] = text_tokenized
    article["article_text_pos"] = text_pos

    comm = article["comments"]
    for c in comm:
        commet_tokenized, comment_pos = process_text(nlp, c["comment_text"])
        c["comment_text_tokenized"] = commet_tokenized
        c["comment_text_pos"] = comment_pos
    return article

#spacy model für language#spacy load english/german,
def process_text(nlp,text):
    text_preprocessed = []
    tags = []
    text = text.replace("\n"," ")
    text = nlp(text)
    for word in text:
        if word.is_stop or word.like_email or word.like_url or word.is_punct or word.like_num:
            continue
        else:
            text_preprocessed.append(word.lemma_.lower())
            tags.append((word.lemma_.lower(),word.tag_))

    return (" ").join(text_preprocessed),tags



if __name__ == '__main__':
    languages = ['german','english']
    RAW_DIR = '../../data/raw_data'
    res_dir = '../../data/tagged'

    for lang in languages:
        if lang == "german":
            nlp = spacy.load("de_core_news_sm")
        elif lang == "english":
            nlp = spacy.load("en_core_web_sm")
        os.makedirs(os.path.join(res_dir, lang), exist_ok=True)
        for fn in os.listdir(os.path.join(RAW_DIR, lang)):
            if fn.endswith('.json'):
                with open(os.path.join(RAW_DIR, lang, fn), 'r', encoding='utf-8') as tf:
                    data = json.load(tf)
                    article_per_ressource = process_resource(nlp,data)
                with open(os.path.join(res_dir,lang,fn), 'w', encoding='utf-8') as outf:
                    json.dump(article_per_ressource, outf)
            print(fn)