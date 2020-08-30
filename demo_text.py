import tensorflow as tf
from tensorflow.keras import layers,Model
import pandas, numpy, string
from sklearn import metrics
from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from bayes import NBClassifier

#加载数据集
data = open('Dataset/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1])

#创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

#将数据集分为训练集和验证集
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label编码为目标变量
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#词语级tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x).A
xvalid_tfidf = tfidf_vect.transform(valid_x).A

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.train(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)

# 特征为词语级别TF-IDF向量的朴素贝叶斯
accuracy = train_model(NBClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)