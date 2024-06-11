import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# Text data
data = """
In the town of Athy one Jeremy Lanigan
Battered away till he hadn't a pound.
His father died and made him a man again
Left him a farm and ten acres of ground.
He gave a grand party for friends and relations
Who didn't forget him when come to the wall,
And if you'll but listen I'll make your eyes glisten
Of the rows and the ructions of Lanigan’s Ball.
Myself to be sure got free invitation,
For all the nice girls and boys I might ask,
And just in a minute both friends and relations
Were dancing round merry as bees round a cask.
Judy O'Daly, that nice little milliner,
She tipped me a wink for to give her a call,
And I soon arrived with Peggy McGilligan
Just in time for Lanigan's Ball.
"""

# Preprocess the text data
data = data.lower()
chars = sorted(set(data))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}
total_chars = len(chars)

# Create character sequences
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i: i + maxlen])
    next_chars.append(data[i + maxlen])
print('Number of sequences:', len(sentences))
# Vectorize sequences
x = np.zeros((len(sentences), maxlen, total_chars), dtype=np.bool_)
y = np.zeros((len(sentences), total_chars), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, total_chars)))
model.add(Dropout(0.2))
model.add(Dense(total_chars, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# Train the model
history = model.fit(x, y, batch_size=128, epochs=20)


# Text generation function
def generate_text(seed_text, next_chars, model, maxlen, char_to_index, index_to_char):
    generated = ''
    sentence = seed_text[-maxlen:]
    for _ in range(next_chars):
        x_pred = np.zeros((1, maxlen, total_chars))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = index_to_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char
    return generated

# Generate new text
seed_text = "in the town of athy"
print(generate_text(seed_text, 100, model, maxlen, char_to_index, index_to_char))
