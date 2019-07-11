import urllib.request
import os
import numpy
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt



url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "data/titanic3.xls"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print(result)

all_df = pd.read_excel(filepath)
cols = ['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
all_df = all_df[cols]
msk = numpy.random.rand(len(all_df))<0.8
print(len(msk))
print(msk)
train_df = all_df[msk]
test_df = all_df[~msk]

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

def PreprocessData(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df, columns=["embarked"])
    ndarray = x_OneHot_df.values
    Label = ndarray[:, 0]
    Features = ndarray[:, 1:]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures,Label

train_Features,train_Label = PreprocessData(train_df)
test_Features,test_Label = PreprocessData(test_df)

model =Sequential()
model.add(Dense(units=40,input_dim=9,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x=train_Features,y=train_Label,validation_split=0.1,epochs=30,batch_size=30,verbose=2)

show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x=test_Features,y=test_Label)
print(scores)

Jack = pd.Series([0,'Jack',3,'male',23,1,0,5,'S'])
Rose = pd.Series([1,'Rose',1,'female',20,1,0,100,'S'])

JR_df = pd.DataFrame([list(Jack),list(Rose)],columns=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked'])
all_df = pd.concat([all_df,JR_df])

all_Features,Label = PreprocessData(all_df)
all_probability = model.predict(all_Features)

pd = all_df
pd.insert(len(all_df.columns),'probability',all_probability)

print(pd[-2:])