from __future__ import division
import numpy as np
from numpy import array
from pickle import dump
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional
from keras import optimizers
from keras.layers import Embedding
from numpy import asarray
'''
filename = 'glove.6B.50d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r',encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd
vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embedding[0])
#Pre-trained GloVe embedding



def np_nearest_neighbor(x):
    xdoty = np.multiply(embedding,x)
    xdoty = np.sum(xdoty,1)
    xlen = np.square(x)
    xlen = np.sum(xlen,0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen,1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen,ylen)
    cosine_similarities = np.divide(xdoty,xlenylen)
    return embedding[np.argmax(cosine_similarities)]
'''



import string
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
def clean(text):
    text = text.lower()
    pr = set(string.printable)
    return filter(lambda x:x in pr,text)
filename = 'republic.txt'
#filename1 = 'Jesus1.txt'
with open(filename,'r') as f:
    file = f.read()
    print(file[:200])
#with open(filename1,'r') as f1:
 #   file1 = f1.read()

def clean_doc(doc):
    doc = doc.replace('--',' ')
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    #stop_words = set(stopwords.words('english'))
    #tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word.lower() for word in tokens]
    return tokens
def save_doc(lines,filename):
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()

finput = clean_doc(file)
print(finput[:200])
'''
fsumm = clean_doc(file1)
print(fsumm)
'''
print(len(finput))
print(len(set(finput)))
length = 50 + 1
sequences = list()
for i in range(length,len(finput)):
    seq = finput[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)
print(len(sequences))
out_file = "file.txt"
save_doc(sequences,out_file)

with open(out_file,'r') as a:
    f1 = a.read()
    lines = f1.split('\n')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)    
vocab_size = len(tokenizer.word_index) + 1000
print(vocab_size)
sequences = array(sequences)
X,y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y,num_classes=vocab_size)
seq_length = X.shape[1]


embeddings_index = dict()
f = open('glove.6B.100d.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

		
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size,100,weights = [embedding_matrix] ,input_length=seq_length))
    model.add(Bidirectional(LSTM(300, return_sequences=True,dropout=0.25,
                                 recurrent_dropout=0.25)))
    model.add(Bidirectional(LSTM(300)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    adam = optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None)
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model
model = define_model(vocab_size,seq_length)
model.fit(X,y,batch_size=128,epochs=100,
          validation_split=0.1)
model.save('model1.h5')
dump(tokenizer,open('tokenizer.pkl','wb'))

#print(y.shape[62,4])
