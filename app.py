# import dependencies
import warnings
warnings.filterwarnings("ignore")
from flask import request , Flask ,render_template
import json
from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential  # model to classify / Lookup
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras.utils import to_categorical
import numpy as np
from os import getenv,walk,path
from keras.models import load_model
from keras import backend as K
APP_ROOT = path.dirname(path.abspath(__file__))
tokenize,encoder,MAX_WORDS,score=0,0,0,0

class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

K.clear_session()

data = read_csv("twitter.txt",sep='\|\~')
data.columns = ['text','label']
columns = data.columns
stopwords = read_csv(path.join(APP_ROOT,'english'))
stopwords.columns = ['stopwords']
stop = set(stopwords['stopwords'].values)
data['text'].apply(lambda x: [item for item in x if item.replace('"', '') not in stop])

MAX_WORDS=2000
EPOCHS = 2
BATCH_SIZE = 64

train_size = int(len(data) * .8)

train_comments = data['text']
train_labels = data['label']
test_comments = data['text'][train_size:]
test_labels = data['label'][train_size:]

tokenize = text.Tokenizer(num_words=MAX_WORDS, char_level=False)
tokenize.fit_on_texts(train_comments)  # only fit on train

x_train = tokenize.texts_to_matrix(train_comments)
x_test = tokenize.texts_to_matrix(test_comments)

encoder =  LabelEncoderExt()
encoder.fit(train_labels)   

y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)
num_classes = np.max(y_train) + 1
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(512, input_shape=(MAX_WORDS,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
print("Model Trained")

print("Saving the model")
# Save the trained model as a pickle string. 
model.save("model.h5")
del y_train,y_test,x_train,x_test,data

# bootstrap the app
app = Flask(__name__)

# set the port dynamically with a default of 3000 for local development
port = int(getenv('PORT', '8080'))

@app.route('/', methods=['POST','GET'])
def default():
    K.clear_session();
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    K.clear_session();
    test_data = [request.args.get("text")]
    df = DataFrame(test_data,columns =['text']) 
    tokenized_posts = tokenize.texts_to_matrix(df['text'])
    response = []
    model = load_model('model.h5')
    model._make_predict_function()
    for i in range(0,len(test_data)):
        prediction = model.predict(np.array([tokenized_posts[i]]))
        predicted_label = encoder.classes_[np.argmax(prediction)]
        response.append({"text":test_data[i],"category":predicted_label})
    return json.dumps(response)

# start the app
if __name__ == '__main__':
    #preprocessing()
    app.run(host='127.0.0.1', port=port, debug=False)