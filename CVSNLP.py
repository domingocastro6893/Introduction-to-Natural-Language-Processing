import csv
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
# Define variables to store sentences and labels
sentences = []
labels = []

# Read data from CSV file
with open('text_emotion.csv', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append(row[0])
        sentence = row[1].lower()

        # Preprocess the sentence
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence, 'html.parser')
        sentence = soup.get_text()
        # Append preprocessed sentence to list
        sentences.append(sentence)

# Split data into training and test subsets
training_size = 28000
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# Tokenize the text
vocab_size = 20000
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Convert text to sequences and pad them
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)

# Tokenize and pad the testing sentences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                               padding=padding_type,
                               truncating=trunc_type)


# Inspect the results
print(training_sequences[0])
print(training_padded[0])
print(word_index)

# Calculate predictions on the testing data
predictions = model.predict(testing_padded)

# Convert predicted probabilities to class labels (e.g., 0 or 1 for binary classification)
predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]

# Print classification report
print(classification_report(testing_labels, predicted_labels))