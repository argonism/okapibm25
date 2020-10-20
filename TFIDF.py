# CJE3で作成したもの

# ---------------------------------------------
# 1. 初期設定
# ---------------------------------------------

### A. ライブラリの読み込み
import os
from janome.tokenizer import Tokenizer
import re
import math

# 分かち書きクラスの初期化
t = Tokenizer()

### B. フォルダやファイル名の設定

# データフォルダの設定
DATA = "data"

# 索引ファイルの指定
INDEX = "index"
index_file = INDEX + "/tf_idf.txt"

### C. 各種辞書オブジェクトの初期化

# 索引語用辞書オブジェクトの初期化
dict_ = {}

# 不要語辞書の初期化
stopwords = {}

# ファイル名保存用辞書オブジェクト
docs = {}

# df用辞書オブジェクトの初期化
df = {}

# idf値用辞書オブジェクト
idf = {}

# ---------------------------------------------
# 2. 不用語削除ルールの定義
# ---------------------------------------------

### A. 正規表現
pattern = re.compile(r"^[　-ー]$")

### B. 不要語リスト
stopwords['という'] = 1
stopwords['にて'] = 1

# ---------------------------------------------
# 3. 文書ファイルの処理
# ---------------------------------------------

# DATAフォルダに含まれるファイルを一つずつ処理
for filename in os.listdir(DATA):

    # ファイルを読み込みモードで開く
    f = open(DATA + '/' + filename, 'r')

    # ファイルを1行ずつ処理
    for line in f:

        ### A. 分かち書き
        tokens = t.tokenize(line)

        # 分かち書きされた語オブジェクトの処理
        for token in tokens:

            ### B. 不用語処理
            # 正規表現
            if pattern.match(token.surface):
                continue

            # 不用語リスト
            if token.surface in stopwords:
                continue

            ### C. 索引語の追加
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

    f.close

# ---------------------------------------------
# 4. 索引語の重み付け
# ---------------------------------------------

### A. 総文書数の算出

# 索引語を一つずつ処理
for word in dict_:

    # 文書を一つずつ処理
    for doc in dict_[word]:

        # もし辞書オブジェクトに文書がないなら追加
        if doc not in docs:
            docs[doc] = 1


# 総文書数（N）を求める
docs_size = len(docs)

### B. 文書頻度の算出

# 索引語を一つずつ処理
for word in dict_:

    # 文書数を取得し、dfに代入
    df[word] = len(dict_[word])


### C. 逆文書頻度の算出

# dfのキー（索引語）を一つずつ処理
for word in df:

    # idf値を計算し、格納
    idf[word] = math.log((docs_size / df[word]) + 1)


# ---------------------------------------------
# 5. 索引ファイルの書き出し
# ---------------------------------------------

# ファイルを書き込みモードで開く
w = open(index_file, "w")

# ソートした索引語を一つずつ処理
for word in sorted(dict_):

    # ソートした文書を一つずつ処理
    for doc in sorted(dict_[word]):

        ### A. tfidf値の算出
        tfidf = dict_[word][doc] * idf[word]

        ### B. 索引語と重みデータの出力
        w.write(word + '\t' + doc + '\t' +
                str(dict_[word][doc]) + '\t' + str(idf[word]) + '\t' + str(tfidf) + "\n")

# ファイルを閉じる
w.close()
