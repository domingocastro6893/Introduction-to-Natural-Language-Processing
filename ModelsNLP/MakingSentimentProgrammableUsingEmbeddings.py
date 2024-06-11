import json
import os
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the path to the JSON file
file_path = "/sarcasm/Sarcasm_Headlines_Dataset_v2.json"

# Initialize lists to store data
all_sentences = []
all_labels = []
all_urls = []

# Open and load the JSON file
with open(file_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        sentence = item['headline'].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence, "html.parser")
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = " ".join(words)
        all_sentences.append(filtered_sentence)
        all_labels.append(item['is_sarcastic'])
        all_urls.append(item['article_link'])

# Split data into training and test sets
training_size = int(len(all_sentences) * 0.8)
training_sentences = all_sentences[:training_size]
testing_sentences = all_sentences[training_size:]
training_labels = all_labels[:training_size]
testing_labels = all_labels[training_size:]

# Tokenize the text
vocab_size = 20000
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert labels to numpy arrays
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
num_epochs = 15
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# Model summary
model.summary()

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_history(history)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(testing_padded, testing_labels, verbose=2)
print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')

# Make predictions on new data
predictions = model.predict(testing_padded[:5])
print(predictions)
print(testing_labels[:5])

# Optional: Print the word index
#print(word_index)


#1. Adjusting the Learning Rate
#adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#model.compile(loss='binary_crossentropy',
#              optimizer=adam, metrics=['accuracy'])

#2 Exploring Vocabulary Size
"""
2. Exploring Vocabulary Size
The vocabulary size refers to the number of unique words the
 model can recognize. If the vocabulary is too large, 
 the model might overfit by learning very rare words that only
  appear in the training data.

Example of Vocabulary Frequency Analysis:

python
Copy code
wc = tokenizer.word_counts
from collections import OrderedDict
newlist = OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True))
print(newlist)
Plotting Word Frequencies:

python
Copy code
import matplotlib.pyplot as plt
xs = []
ys = []
curr_x = 1
for item in newlist:
    xs.append(curr_x)
    curr_x += 1
    ys.append(newlist[item])
plt.plot(xs, ys)
plt.axis([300, 10000, 0, 100])
plt.show()
"""

#3. Exploring Embedding Dimensions
"""
#3. Exploring Embedding Dimensions
Embedding dimensions refer to the size of the vector space in which words are represented. A larger dimension might capture more nuances of the words' meanings, but it can also lead to overfitting if the dimensionality is too high relative to the number of words.

Adjusting Embedding Dimensions:

python
Copy code
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(2000, 7),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
Using a smaller embedding dimension (like 7) can make the model more efficient and reduce overfitting.

4. Exploring the Model Architecture
The architecture of the model, including the number and size of layers, can also influence overfitting. Simplifying the model (e.g., using fewer neurons in dense layers) can help in reducing overfitting.

Simplified Model Architecture:

python
Copy code
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(2000, 7),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])"""
