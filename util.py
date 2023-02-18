import nltk
import csv
import re
from bs4 import BeautifulSoup
import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn import feature_extraction
import torch.nn as nn
import torch

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def read_text_data(fname):
    txtdata = []
    revid   = []
    classes = []
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            # get the text
            txtdata.append(cleanText(row[0]))
            revid.append(row[1])
            # get the class (convert to integer)
            if len(row)>2:
                classes.append(int(row[2]))
    
    if (len(classes)>0) and (len(txtdata) != len(revid)):        
        warn.error("mismatched length!")
    
    return (txtdata, revid, classes)

def write_csv_kaggle_sub(fname, Y):
    # fname = file name
    # Y is a list/array with class entries
    
    # header
    tmp = [['Id', 'Prediction']]
    
    # add ID numbers for each Y
    for (i,y) in enumerate(Y):
        tmp2 = [(i+1), y]
        tmp.append(tmp2)
        
    # write CSV file
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(tmp)

# nltk.download("punkt")  # you may need to download when you never use this nltk punkt
from gensim.models.doc2vec import Doc2Vec
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def txt2tagdoc(txt, tags):
    tagdoc = []
    for i, (text, tag) in enumerate(zip(txt, tags)):
        words = tokenize_text(text)
        document = TaggedDocument(words, [tag])
        tagdoc.append(document)
    return tagdoc


def gen_feature(size, traintxt, testtxt):
    cntvect = feature_extraction.text.CountVectorizer(stop_words='english', max_features=size if size!= None else None)

    cntvect.fit(traintxt)

    traindata_trans = cntvect.transform(traintxt)
    testdata_trans = cntvect.transform(testtxt)

    tf_trans = feature_extraction.text.TfidfTransformer(use_idf=True, norm='l1')

    trainXtfidf = tf_trans.fit_transform(traindata_trans)

    testXtfidf = tf_trans.fit_transform(testdata_trans)
    return trainXtfidf, testXtfidf

def vec_for_learning(model, tagged_docs):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, epochs=20)) for doc in tagged_docs])
    return targets, regressors

class GetTupleFirst(nn.Module):
    def __init__(self, *args):
        super(GetTupleFirst, self).__init__()
        self.index = args
        # self.linear = nn.Linear(1, 1)
    def forward(self, input):
        return input[0]

class Lstm_Bi(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size,
                 biFlag, dropout=0.5):
        super(Lstm_Bi, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        if (biFlag):
            self.bi_num = 2
        else:
            self.bi_num = 1
        if biFlag:
            self.linear = nn.Sequential(
                nn.Linear(self.bi_num*hidden_dim, hidden_dim),
                nn.Dropout(),
                nn.GELU(),
            )
        self.biFlag = biFlag
        if num_layers>1:
            self.lstm = nn.ModuleList()
            for i in range(num_layers):
                self.lstm.append(nn.Sequential(
                    nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=biFlag),
                    GetTupleFirst(),
                    nn.LayerNorm(hidden_dim),
                ))
        else:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                                num_layers=num_layers, batch_first=True,
                                dropout=dropout, bidirectional=biFlag)
        self.hidden = (torch.randn(self.num_layers * self.bi_num,
                                   batch_size, self.hidden_dim),
                       torch.randn(self.num_layers * self.bi_num,
                                   batch_size, self.hidden_dim))
        self.norm = nn.LayerNorm([self.bi_num*hidden_dim])
    
    def forward(self, input, model_args=None, attention_mask=None):
        if self.num_layers>1:
            for layer in self.lstm:
                input = layer(input)
            out = nn.Dropout()(input)
        else:
            out, (h1, c1) = self.lstm(input)
            out = self.norm(out)
            out = nn.Dropout()(out)
        if self.bi_num==2:
            out = self.linear(out)
        out = out + input
        return out