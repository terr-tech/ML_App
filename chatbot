#Chatbot Coursework project
#This chatbot utilizes Facebook Research dataset Babi
#The intent for this piece of code is to test out simple neural network building
#to construct a natural language processing chatbot.
#The end result may be interesting as users can enter a story, the bot will pick
#it up, then user can ask questions for the chatbot to spit out a binary answer.
#YES/NO

#PART 1
#Data processing
import pickle
import numpy as np

#load files
with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)

with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

#combining datasets
#create all data
all_data = test_data + train_data
set(train_data[0][0])

#set the vocab used for later
vocab = set()
for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

#adding in yes/no into the vocab
vocab.add('yes')
vocab.add('no')


vocab_len = len(vocab) + 1
#print(vocab_len)

#longest story
all_story_lens = [len(data[0]) for data in all_data]
max_story_len = max(all_story_lens)
#print(max_story_len)

max_question_len = max([len(data[1]) for data in all_data])
#Print (max_question_len)


#PART 2
#after we explore the data, we vectorize it.
#vectorization:
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#uses a TensorFlow backend

tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)
tokenizer.word_index

#initiate training sets
train_story_text = []
train_question_text = []
train_answers = []

#read-in
for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)
#print (train_story_text)

#converting to index calls
train_story_seq = tokenizer.texts_to_sequences(train_story_text)
#print (train_story_seq)


#Functionalize Vectorization
#INPUT:

#data: consisting of Stories,Queries,and Answers
#word_index: word index dictionary from tokenizer
#max_story_len: length of the longest story (used for pad_sequences function)
#max_question_len: length of the longest question (used for pad_sequences function)


#OUTPUT:

#Vectorizes the stories,questions, and answers into padded sequences. We first
#loop for every story, query , and answer in the data. Then we convert the raw
#words to an word index value. Then we append each set to their appropriate
#output list. Then once we have converted the words to numbers, we pad the
#sequences so they are all of equal length.

#Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)

def vectorize_stories(data, word_index = tokenizer.word_index,
                        max_story_len = max_story_len,
                         max_question_len = max_question_len):
    #STORIES = X
    X = []
    #QUESTIONS = Xq
    Xq = []
    #Y CORRECT ANSWER(YES/NO)
    Y = []

    for story, query, answer in data:

        # for each story
        # [23, 14, ....]
        #word index for every word in story
        x = [word_index[word.lower()] for word in story]
        #word index for every word in query
        xq = [word_index[word.lower()] for word in query]
        #use +1 as we want to reserve index 0
        y = np.zeros(len(word_index)+1)

        #yes/no
        y[word_index[answer]] = 1

        #appending to the list
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen = max_story_len),
            pad_sequences(Xq,maxlen=max_question_len), np.array(Y))


#training data:
inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

#print(inputs_test)
#print(answers_test)

tokenizer.word_index['yes']
tokenizer.word_index['no']
#print (sum(answers_test))


#PART 3
#This is the best part, as we start building the neural network
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM

#placeholder shape = (max_story_len, batch_size)
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

#vocab_len
vocab_size = len(vocab) + 1

#Input Encoder M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_size, output_dim = 64))
input_encoder_m.add(Dropout(0.3))
#OUTPUT
#(samples, story_maxLen, embedding_dim)

#Input Encoder M
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_size,
                    output_dim = max_question_len))
input_encoder_c.add(Dropout(0.3))
#OUTPUT
#(samples, story_maxLen, embedding_dim)


question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_size,
                               output_dim = 64, input_length = max_question_len))
question_encoder.add(Dropout(0.3))
#(samples, query_maxLen, embedding_dim)

#ENCODED <--- ENCODER(INPUT)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

#creation of match
match = dot([input_encoded_m, question_encoded],axes = (2,2))
match = Activation('softmax')(match)

#creation of response
response = add([match,input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response,question_encoded])
#print(answer)

answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)

answer = Activation('softmax')(answer)

#create the model
model = Model([input_sequence, question], answer)

model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

#Training the model aka fitting the model
history = model.fit([inputs_train, queries_train], answers_train,
                    batch_size=32,epochs=120,validation_data=(
                    [inputs_test, queries_test],answers_test))


#PART 4
#Evaluate it
model.load_weights(filename)
pred_results = model.predict(([inputs_test, queries_test]))

story =' '.join(word for word in test_data[0][0])
#print(story)

query = ' '.join(word for word in test_data[0][1])
#print(query)

print("True Test Answer from Data is:",test_data[0][2])


#Generate prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])



#PART 5
#Test out your own story, query and answer
# Note the whitespace of the periods
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()
my_question = "Is the football in the garden ?"
my_question.split()
mydata = [(my_story.split(),my_question.split(),'yes')]
my_story,my_ques,my_ans = vectorize_stories(mydata)
pred_results = model.predict(([ my_story, my_ques]))

#Generate prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
