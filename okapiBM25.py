# A. ライブラリの読み込み
import os
from janome.tokenizer import Tokenizer
import re
import math


class OkapiBM25:

    def __init__(self, document_path):
        self.t = Tokenizer()
        self.tc = self.set_termcount(document_path)
        self.docs = self.set_docs()

        self.docs_size = len(self.docs)
        self.dl = self.set_dl()
        # self.tf = self.calc_tf()
        # self.idf = self.calc_idf()
        # self.avgdl = self.calc_avgdl()

        self.k1 = 2.0
        self.b = 7.5

    def set_termcount(self, dir_path):
        dict_ = {}
        # DATAフォルダに含まれるファイルを一つずつ処理
        for filename in os.listdir(dir_path):

            # ファイルを読み込みモードで開く
            with open(os.path.join(dir_path, filename), 'r') as f:
                # ファイルを1行ずつ処理
                for line in f:
                    dict_ = self.get_termcount(line, filename, dict_)
        return dict_

    def get_termcount(self, text, filename, tc_dict=None):
        dict_ = tc_dict if tc_dict else {}
        tokens = self.t.tokenize(text)
        # 分かち書きされた語オブジェクトの処理
        for token in tokens:

            # C. 索引語の追加（とその単語のその文書内での出現回数）
            # dict_[単語][文書] = その単語の出現回数(int)
            if token.surface in dict_:
                # もし、そのキーの値が参照する無名辞書オブジェクトにファイル名をキーとするレコードが存在していれば
                if filename in dict_[token.surface]:
                    dict_[token.surface][filename] += 1
                # さもなくば
                else:
                    dict_[token.surface][filename] = 1
            # さもなくば
            else:
                dict_[token.surface] = {}
                dict_[token.surface][filename] = 1

        return dict_

    def set_docs(self):
        docs = {}
        # 索引語を一つずつ処理
        for word in self.tc:
            # 文書を一つずつ処理
            for doc in self.tc[word]:
                # もし辞書オブジェクトに文書がないなら追加
                if doc not in docs:
                    docs[doc] = 1

        return docs

    def calc_tf(self, word, doc, dl):
        return self.tc[word][doc] / dl

    def calc_idf(self, word, docs_size):
        df = len(self.tc[word])
        return math.log(
            (docs_size - df + 0.5) / (df + 0.5)
        )

    def set_dl(self):
        dl = {}
        for word in self.tc:
            # 文書を一つずつ処理
            for doc in self.tc[word]:
                # もし辞書オブジェクトに文書がないなら追加
                if doc not in dl:
                    dl[doc] = self.tc[word][doc]
                else:
                    dl[doc] += self.tc[word][doc]
        return dl

    def calc_avgdl(self, dls):
        return sum(dls) / len(dls)

    def get_scores(self):
        score = {}
        avgdl = self.calc_avgdl(self.dl.values())
        for word in self.tc:
            if word not in score:
                score[word] = {}
            for doc in self.tc[word]:
                idf = self.calc_idf(word, self.docs_size)
                # 半分以上に属する語は一般的なものとみなして、無視する
                # (半分以上に属する語 = idf < 0)
                idf = idf if idf > 0 else 0
                dl = self.dl[doc]
                tf = self.calc_tf(word, doc, dl)
                k1 = self.k1
                b = self.b

                score[word][doc] = self.calc_combined_weight(
                    idf, tf, dl, k1, b, avgdl)

        return score

    def search(self, query):
        q_tc = self.get_termcount(query, 'query')
        print(q_tc)
        result = {}
        avgdl = self.calc_avgdl(self.dl.values())
        for word in q_tc:
            if word not in self.tc:
                continue  # その単語を含む文書が存在しないとき、tf = 0 -> score = 0
            for doc in self.tc[word]:
                idf = self.calc_idf(word, self.docs_size)
                idf = idf if idf > 0 else 0
                dl = self.dl[doc]
                tf = self.calc_tf(word, doc, dl)
                k1 = self.k1
                b = self.b

                score = self.calc_combined_weight(
                    idf, tf, dl, k1, b, avgdl)

                if doc in result:
                    result[doc] += score
                else:
                    result[doc] = score
        return result

    def calc_combined_weight(self, idf, tf, dl, k1, b, avgdl):
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dl / avgdl))
        return idf * (numerator / denominator)


def export(path, scores):
    index_file_path = "index"
    index_file_name = "okapi_bm25.txt"
    with open(os.path.join(script_path, index_file_path, index_file_name), 'w') as file:
        for word in scores:
            if len([scores[word][doc] for doc in scores[word] if scores[word][doc] > 0]) == 0:
                continue
            file.write(word + '\n')
            for doc in scores[word]:
                if scores[word][doc] <= 0:
                    continue
                file.write("    {1} ->  {2}\n".format(word,
                                                      doc, scores[word][doc]))


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = "data"
    okapi = OkapiBM25(os.path.join(script_path, data_path))
    scores = okapi.get_scores()

    index_file_path = "index"
    index_file_name = "okapi_bm25.txt"
    export(os.path.join(script_path, index_file_path, index_file_name), scores)

    result = okapi.search('池の前')
    print(result)
    #
    # the result will be
    #
    # {
    #   'sample_doc8.txt': 0.1906291414340978,
    #   'sample_doc9.txt': 0.006381205101560838,
    #   'sample_doc10.txt': 0.0,
    #   'sample_doc2.txt': -0.0,
    #   'sample_doc3.txt': 0.0,
    #   'sample_doc4.txt': -0.0,
    #   'sample_doc5.txt': -0.0,
    #   'sample_doc7.txt': 0.0,
    #   'sample_doc6.txt': 0.0
    # }
    #
