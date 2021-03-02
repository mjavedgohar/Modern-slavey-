Modern slavey Statements Detection 


A. What methodology do you propose to assess the quality of text extracted under
column ‘TEXT’? This text is extracted from the URLs in column ‘Answer Link’. ?

The main objective of this activity is to detect the modern slavery statements whether the company has a due diligence process in place to monitor these risks through continuous engagement with suppliers.
After manually going through the provided data set, it is noticed that data is not ladled properly. some of the documents don't have the labels while some with other than the metric's options i.e., audits of suppliers (self- reporting), audits of suppliers (independent), on-site visits (self- reporting), on-site visits (independent), in development, no. Therefore, I selected only those documents (in function "load_doc" at first conditional statement at line # 62) that are properly labelled (as per defined metric’s).
Secondly, It is also observed that there are some irrelevant / extra statements in the text (e.g., headings, company introduction). So, I defined a function "selected_sentences", that takes the file as input and extract only those sentences that are talking about modern slavery or due diligence monitoring process by considering the keywords. These selected statements are considered for the rest of the process.



B. Present the code of the solutions developed for this metric and interpret your results. Ensure that each section of the solution is well described and documented.
In this sample solution, I tried to train the CNN model for the detection of modern slavery statements. I followed the following steps;
1.	Loaded only those documents that are properly labelled as per metric’s using the function load_doc. From each document only those statements/sentences extracted in which it is discussed about the modern slavery and/or monitoring process. 
2.	Create_vocab() function is to pre-process (removing stop words, digits, or words with less then or equal to one char ) the documents and create vocabulary. I got the clean_docs and vocab from this function
3.	 Add_doc_to_vocab() is to activate/call the load_doc() and create_vocab() functions
4.	Vocab_select() function is to add only those words in documents text that are in the vocabulary. This will help to encode docs efficiently.
5.	Lines# 152 to 162 are to encode the documents and padding the sentences.
6.	Lines# 165 to 185 are to encode labels. In this encoding, each label is converted to the vector of length 6 because there are 6 metric’s i.e., audits of suppliers (self- reporting), audits of suppliers (independent), on-site visits (self- reporting), on-site visits (independent), in development, & no. If any of these metrics occurs in label that bit is turned on (1). As a result each encoded label is a list of o’s and 1’s e.g. [0,1,0,1,0,0].
7.	At line # 197, I divided the training and testing data.
8.	Rest of the lines are to define model, train and validate it. Model’s summary is as under


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 2080, 100)         978300
_________________________________________________________________
conv1d (Conv1D)              (None, 2073, 32)          25632
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 1036, 32)          0
_________________________________________________________________
flatten (Flatten)            (None, 33152)             0
_________________________________________________________________
dense (Dense)                (None, 20)                663060
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 126
=================================================================
Total params: 1,667,118
Trainable params: 1,667,118
Non-trainable params: 0
_________________________________________________________________


C. How do you assess the quality of your results? What is the accuracy of your model? How do you interpret your results? How does your model perform for each metric option? What are the challenges? What would you recommend to do to improve your initial results?
I will assess the trained model on data collected from different organization. The accuracy of my trained model is: Training=87% & validation=43%. However, I can optimize the model for more accurate analysis. The model can perform well for the metric options. I am also interested to use Transformer model to optimise the efficiency. In future, I’ll consider the following steps to improve the results (document level).
a)	I’ll train and apply a neural network model for the summarization of documents (improvement in sentences selection step)
b)	I’ll try some other NN models depending on the data to make it more efficient.

We can evaluate the sentences/paragraphs for the metric’s. For this activity, I’ll label and train model on sentence/paragraph level.





D. How would you design and deploy the Project API to align your solution with the
WikiRate platform?
I would like to train a NN model and deploy it as API with a user interface so that it can be used to analyse the documents.



'Explain the prediction by pointing to the part of the text that describes each option of the metric that was found (could be at the level of sentence or paragraph).'? 
I trained the model by considering the complete documents. In next step, I’ll train a model for the sentences/paragraphs to predict the metric. I think its easier than document level. For this step;
•	Firstly, I’ll develop an automated labelling system for sentences based on keywords. 
•	Then train a model for the evaluation of documents.



