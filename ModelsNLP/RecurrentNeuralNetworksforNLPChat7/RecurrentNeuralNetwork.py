import tensorflow as tf
import json

# Define the path to the JSON file
file_path = "C:\\Users\\14014\\kenzie\\NaturalLanguagerocessingTF\\sarcasm\\Sarcasm_Headlines_Dataset_v2.json"


# Initialize lists to store data
all_sentences = []
all_labels = []

# Open and load the JSON file
with open(file_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        sentence = item['headline'].lower()
        all_sentences.append(sentence)
        all_labels.append(item['is_sarcastic'])

# Split data into training and test sets
training_size = int(len(all_sentences) * 0.8)
training_sentences = all_sentences[:training_size]
testing_sentences = all_sentences[training_size:]
training_labels = all_labels[:training_size]
testing_labels = all_labels[training_size:]

# Tokenize the text
vocab_size = 20000
embedding_dim = 16


# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define the optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Print model summary
model.summary()
