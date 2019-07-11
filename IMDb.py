import urllib.request
import os
import tarfile
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN,LSTM
import re

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print(result)

if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz","r:gz")
    result = tfile.extractall('data/')

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path+f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path+f]

    print('read',filetype,'files:',len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input :
            all_texts += [rm_tags("".join(file_input.readlines()))]

    return all_labels,all_texts

SentimentDict = {1:'正面的',0:'负面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('label真实值：',SentimentDict[y_test[i]])
    print('预测值：',SentimentDict[predict_classes[i]])

def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq,maxlen=380)
    predict_result = model.predict_classes(pad_input_seq)
    print(predict_result)
    print(predict_result[0])
    print(predict_result[0][0])
    print(SentimentDict[predict_result[0][0]])

y_train,train_text = read_files("train")
y_test,test_text = read_files("test")

token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)

print(token.document_count)
print("ppp")

a = token.word_index
print(type(a))
print(len(a))
print(a['the'],a['and'],a['a'])

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
print(train_text[0])
print(x_train_seq[0])

x_train = sequence.pad_sequences(x_train_seq,maxlen=380)
x_test = sequence.pad_sequences(x_test_seq,maxlen=380)

model = Sequential()
model.add(Embedding(output_dim=32,input_dim=3800,input_length=380))
model.add(Dropout(0.2))

#model.add(Flatten())
#model.add(SimpleRNN(units=16))
model.add(LSTM(32))

model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1,activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=2,validation_split=0.2)

scores = model.evaluate(x_test,y_test,verbose=1)
print(scores)

predict = model.predict_classes(x_test)
predict_classes = predict.reshape(-1)
print(predict_classes[:10])

predict_review('''Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting, because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.''')

