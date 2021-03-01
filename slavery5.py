import csv
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout



 
# selecting only those lines that contain keywords (Summarization)
documents=[]

def selected_sentences(mfile):
    
    is_added=False
    key_words=["due diligence", "human rights due diligence", "planning to implement", "continuous improvement programs", "audit of suppliers", "continuously engaging with suppliers", 
            "engagement", "workers", "trade unions", "on-site visits", "visit", "audits of suppliers", "audits", "audit", "monitor", "third party", "independent", "verification",
                "unannounced", "external", "reviewing our progress", "reviewed progress", "internal audit", "supports the elimination", "human rights working group", "address", "assess",
                "assess modern slavery", "assessment", "monitoring", "took action", "managing", "mitigation", "intervention", "action plan", "identification"]


    
    selected_sentences=[]    
    
    
    text=mfile.splitlines()
    
    # extracting sentences from document based on the keywords
    for row in text:
        if len(row)> 2 and not row[0].isdigit() and len(row.split())>5 and any(x.lower() in row.lower() for x in key_words):  # no heading select and selecting sentences based on keywords
            selected_sentences.append(row)
    #selected_text=''    
    if len(selected_sentences)>=3: # file with more than 3 lines
        selected_text=' '.join([str(lines) for lines in selected_sentences])
        #print(selected_text)
        documents.append(selected_text)
        is_added=True    
                
    return is_added



# load doc into memory
train_labels=[]

def load_doc(filename):
    
    metrics_option=["audits of suppliers", "on-site visits", "in development", "no"]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #selection only those documents that are labeled properly (as per matric's)
            if len(row[6])>20 and (any(x in row[7].lower() for x in metrics_option[:-1]) or row[7].lower()=="no"): 
                                
                if selected_sentences(row[6].lower()):
                    #documents.append(_sentences(row[6].lower())) 
                    train_labels.append(row[7].lower())
                
                
    
	
# define vocablary
vocab = Counter()
clean_docs=[]
# turn a doc into clean tokens
def create_vocab(documents):
    
    for doc in documents:
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        clean_docs.append(tokens)
        # update counts
        vocab.update(tokens)

    



# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	load_doc(filename)
	# clean doc
	create_vocab(documents)
	# update counts
	#vocab.update(tokens)




 


# add all docs to vocab
add_doc_to_vocab('labeled_dataset.csv', vocab)

# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))




# selecting those words for a document that are in the vocabulary  
def vocab_select(doc):
    
    tokens = [w for w in doc if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

train_docs=[]
for doc in clean_docs:
    # clean doc
    tokens = vocab_select(doc)
    # add to list
    train_docs.append(tokens)


# printing the number of training files and lables
print(f"train data={len(train_docs)}")
print(f"train labels={len(train_labels)}") 




# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)

# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# preparing/encoding the labels for training model
metrics=["audits of suppliers (self- reporting)", "audits of suppliers (independent)", "on-site visits (self- reporting)", "on-site visits (independent)", "in development", "no"]
metrics=sorted(metrics)

out_empty = [0 for _ in range(len(metrics))]



#print(metrics)
# this loop is to encode each label. If label contains a metric the replace it with one and rest to 0's. each encoded labels is as e.g., [0,1,0,0,0,1]
ytrain=[]
# label indexing
for train in train_labels:
    output_row=out_empty[:]
    labels=train.split(',')
    #print(labels)
    #print(train)
    for label in labels:
        output_row[metrics.index(label.strip())] = 1
    ytrain.append(output_row)


# counting number of metrics for output layer in the model
n_outputs=len(ytrain[0])
# converting the data to numpy array for model training
Xtrain=array(Xtrain)
ytrain=array(ytrain)
print(f"train data={len(Xtrain)}")
print(f"train labels={len(ytrain)}") 

# dividing the training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(Xtrain, ytrain, test_size=0.1, random_state=0)

print(f"train_x={len(x_train)}")
print(f"train_y={len(y_train)}")
print(f"test_x={len(x_test)}")
print(f"test_y={len(y_test)}")

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model for training
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# fit network
history=model.fit(x_train, y_train, epochs=15, verbose=1) #, validation_data=(x_test, y_test))

model.save("sentiment.model")

# evaluate trained model
val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy: %f' % (val_acc*100))



