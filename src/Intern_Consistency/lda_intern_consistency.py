from src.models.topic_models import TopicModel
from src.helper import pkl_list
import matplotlib.pyplot as plt
from src.Intern_Consistency.hLda_jenson_shannon import jensen_shannon
import numpy as np
import pandas as pd
from  scipy.stats import entropy

def plot_alphas_per_model(model_pkl):
    if "lda" not in model_pkl:
        return
    else:
        topicmodel = TopicModel.load(model_pkl)
        alphas = topicmodel.model.alpha
        indices = np.argsort(alphas)[::-1]
        sorted_a = alphas[indices]

        plt.bar(range(0, len(sorted_a)), sorted_a, 0.3)
        plt.title("Alphas for {}".format(model_pkl[:(len(model_pkl))-4]))
        plt.xlabel('Topics')
        plt.ylabel('Alpha')
        #plt.xticks(range(0, len(sorted_a)),indices, rotation=45)
        plt.show()
        #plt.savefig("test.png",bbox_inches = "tight")
        print(model_pkl)
    return topicmodel,alphas

def plot_phi_matrix(model_pkl):
    topicmodel = TopicModel.load(model_pkl)
    ttm = topicmodel.get_topic_token_matrix()

    matrix = np.zeros((len(ttm), len(ttm)))
    topic_num1 = []
    for i, tt in enumerate(ttm):
        topic_num1.append("Topic " + str(i))
        for j, t in enumerate(ttm):
            matrix[i, j] = 1 - jensen_shannon(tt, t)
    df = pd.DataFrame(matrix, columns=topic_num1, index=topic_num1)
    return df

def plot_sorted_phi_matrix(df):
    df_len = len(df)
    matrix = df.values
    sort_arg = np.argsort(matrix, axis=0)[::-1]
    sorted_v = np.sort(matrix, axis=0)[::-1]

    result = []
    for c in range(0, df_len):
        c_list = []
        for r in range(0, df_len):
            t = "Topic " + str(sort_arg[r, c])
            v = round(sorted_v[r, c],3)
            c_list.append((t, v))
        result.append(c_list)
    topic_label = ["Topic " + str(i) for i in range(0, df_len)]
    df2 = pd.DataFrame(result, index=topic_label)
    return df2

def get_entropy(model_pkl):
    topicmodel = TopicModel.load(model_pkl)
    ttm = topicmodel.get_topic_token_matrix()
    res = []
    for i in range(0,len(ttm)):
        row = ttm[i,:]
        entropy_per_row = entropy(row)
        res.append(entropy_per_row)
    return res

def plot_entropy(model_pkl, sorted = True):
    entropy = get_entropy(model_pkl)
    entropy = np.array(entropy)
    if sorted:
        indices = np.argsort(entropy)
        print(indices)
        print(type(indices))
        sorted_e = entropy[indices]
        plt.bar(range(0, len(sorted_e)), sorted_e, 0.3)
    else:
        plt.bar(range(0, len(entropy)), entropy, 0.3)
        plt.title("Entropy for {}".format(model_pkl[:(len(model_pkl)) - 4]))
        plt.xlabel('Topics')
        plt.ylabel('Entropy')
        #plt.xticks(range(0, len(entropy)),range(0, len(entropy)), rotation=45)
        plt.show()
        # plt.savefig("test.png",bbox_inches = "tight")


if __name__ == "__main__":
    #plot_alphas_per_model(pkl_list[0])
    #print(pkl_list)
    t140 = "intern_consistency/topic_models/lda_german_editorial_articles_140.pkl"
    plot_entropy(t140)