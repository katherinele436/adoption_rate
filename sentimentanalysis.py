import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
#from sklearn.linear_model import LogisticRegression

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, SimpleRNN


'''
This function extracts the IDs of the dogs and the target numbers to a new fille "data1.csv"
'''
def getDogs():
    f = open("train.csv", "r", encoding="utf8")
    new = open("data.csv", "w", encoding="utf8")
    line = f.readline()
    line = f.readline()
    while(line is not None and line != ""):
        line = line.split(",")
        new.write(str(line[-3]) + "," + str(line[-1][0:-1]) + "," + str(" ".join(line[20:len(line) - 3]).strip("\"")) + "\n")
        line = f.readline()
    f.close()
    new.close()

def createDTM(messages):
    #remove stop words listed at the top
    stop = ['ourselves', 'hers', 'between',
     'yourself', 'but', 'again', 'there',
     'about', 'once', 'during', 'out', 'very',
     'having', 'with', 'they', 'own', 'an', 'be',
     'some', 'for', 'do', 'its', 'yours', 'such',
     'into', 'of', 'most', 'itself', 'other', 'off',
     'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
     'each', 'the', 'themselves', 'until', 'below',
     'are', 'we', 'these', 'your', 'his', 'through',
     'don', 'nor', 'me', 'were', 'her', 'more',
     'himself', 'this', 'down', 'should', 'our',
     'their', 'while', 'above', 'both', 'up', 'to',
     'ours', 'had', 'she', 'all', 'no', 'when', 'at',
     'any', 'before', 'them', 'same', 'and', 'been',
     'have', 'in', 'will', 'on', 'does', 'yourselves',
     'then', 'that', 'because', 'what', 'over', 'why',
     'so', 'can', 'did', 'not', 'now', 'under', 'he',
     'you', 'herself', 'has', 'just', 'where', 'too',
     'only', 'myself', 'which', 'those', 'i', 'after',
     'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
     'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    vect = TfidfVectorizer(min_df = 50,stop_words = stop)
    dtm = vect.fit_transform(messages) # create DTM
    
    # create pandas dataframe of DTM
    return pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
        

def getWordArrays():
    lis = []
    dat = []
    data = open("data.csv", "r", encoding="utf8")
    for line in data:
        line = line.split(',')
        dat.append(line[0:2])
        lis.append(" ".join(line[2:]).strip())
    data.close()
    return lis, dat

def rnn(X_train,y_train,X_test, y_test):
    inputDim = len(X_train)
    max_words = len(X_train[1])
    model = Sequential()
    model.add(Embedding(output_dim=199, input_dim = inputDim, input_length=max_words))
    model.add(SimpleRNN(5, input_dim = 199,
                    activation = "relu",
                    use_bias = False,
                    input_shape = (max_words,inputDim)))
    #model.add(LSTM(199))
    #model.add(Dense(5,input_dim = inputDim, activation='relu', ))
    #model.add(Dense(5, activation='relu'))
    model.compile(loss='categorical_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
    batch_size = int(len(X_train) / 5)
    num_epochs = 7
    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])

def main(): 
    messages, target = getWordArrays()
    messages = createDTM(messages)
    train_des, test_des, train_lbl, test_lbl = train_test_split(messages,target, test_size = 1 / 7.0, random_state=0)

    # Make an instance of the Model
    pca = PCA(0.8)
    #retaining .8 variance reduces the number of attributes from around 360 to 199
    pca.fit(messages)
    messages = pca.transform(messages)
    train_des = pca.transform(train_des)
    test_des = pca.transform(test_des)

    #
    encoder = LabelEncoder()
    print(train_lbl[1][1])
    train_lbl = [int(row[1]) for row in train_lbl]
    test_lbl = [int(row[1]) for row in test_lbl]
    encoder.fit([int(row[1]) for row in target])
    encoded_Y = encoder.transform(train_lbl)
    encoded_Y1 = encoder.transform(test_lbl)

    # convert integers to dummy variables (i.e. one hot encoded)
    train_y = np_utils.to_categorical(encoded_Y)
    test_y = np_utils.to_categorical(encoded_Y1)

    print(pca.n_components_)
    rnn(train_des, train_y,test_des,test_y)

    #use logistic regression to test the accuracy of the pca
    # logisticRegr = LogisticRegression(solver = 'lbfgs')
    # logisticRegr.fit(train_des, [row[1] for row in train_lbl])
    # logisticRegr.predict(test_des)
    # print(logisticRegr.score(test_des, [row[1] for row in test_lbl]))


getDogs()
#getscentences()
main()

'''
results with logistical regression
starting att - 389
for 99% retetion of variance - accuracy, 0.31538 - number of attributes, 389
for 95% retetion of variance - accuracy, 0.31730 - number of attributes, 317
for 90% retetion of variance - accuracy, 0.32884 - number of attributes, 269
for 85% retetion of variance - accuracy, 0.33653 - number of attributes, 231
for 80% retetion of variance - accuracy, 0.35384 - number of attributes, 199
for 75% retetion of variance - accuracy, 0.34615 - number of attributes, 172
for 70% retetion of variance - accuracy, 0.34230 - number of attributes, 148
'''

'''
sources
- https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
- https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/
- https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e
'''