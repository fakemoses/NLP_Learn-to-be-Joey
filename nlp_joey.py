import numpy as np
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

#read json file
with open("tmp/joey.json", "r",encoding='utf-8') as f:
    datastore = json.load(f)

joey_messages = []
#iterate through the json file
for item in datastore:
    if item == "messages":
        for message in datastore[item]:
            if(message["author"]["name"] == "Linus"):
                joey_messages.append(message["content"])

#filter out anything that contain @ in the list
joey_messages = [message for message in joey_messages if "@" not in message]

#find words between colons in the lists of sentences and save in another list
joey_messages_semicolon = []
for message in joey_messages:
    if(":" in message):
        joey_messages_semicolon.append(message.split(":")[1])

#remove any links and duplicates from the list
joey_messages_semicolon = [message for message in joey_messages_semicolon if "www" not in message]
joey_messages_semicolon = [message for message in joey_messages_semicolon if ".com" not in message]
joey_messages_semicolon = [message for message in joey_messages_semicolon if ".tv" not in message]

#remove anything with more than one word minimum 3 chars and only alphabets in the list
joey_messages_semicolon = [message for message in joey_messages_semicolon if len(message.split()) == 1]
joey_messages_semicolon = [message for message in joey_messages_semicolon if len(message) > 3]
joey_messages_semicolon = [message for message in joey_messages_semicolon if message.isalpha()]
joey_messages_semicolon = list(set(joey_messages_semicolon))

#save joey_messages_semicolon to a file
with open("tmp/joey_emotes.txt", "w",encoding='utf-8') as f:
    for message in joey_messages_semicolon:
        f.write(message + "\n")

#remove colons in joey_messages list
joey_messages = [message for message in joey_messages if ":" not in message]

#LSTM part
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>" , lower=False)
corpuselist = tokenizer.fit_on_texts(joey_messages)
total_words = len(tokenizer.word_index) + 1

#print(total_words)
#print(tokenizer.word_index)

inputsequences = []
for line in joey_messages:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        inputsequences.append(n_gram_sequence)

#pad sequences
max_sequence_len = max([len(x) for x in inputsequences])
inputsequences = np.array(pad_sequences(inputsequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = inputsequences[:,:-1],inputsequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# create model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 25, input_length=max_sequence_len-1),
    tf.keras.layers.LSTM(25, return_sequences=True),
    tf.keras.layers.LSTM(25),
    tf.keras.layers.Dense(total_words, activation='softmax')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])

#early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = model.fit(xs, ys, epochs=100, verbose=1, callbacks=[early_stopping])

#save the model
model.save("tmp/joey_model.h5")

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')

#testing
seed_text = "moses you should"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)