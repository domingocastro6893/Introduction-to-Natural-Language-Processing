import json
import os
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the directory containing the JSON files
directory = "C:\\Users\\14014\\kenzie\\NaturalLanguagerocessingTF\\sarcasm\\Sarcasm_Headlines_Dataset_v2.json"

# Initialize lists to store data
all_sentences = []
all_labels = []
all_urls = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            datastore = json.load(f)
            for item in datastore:
                sentence = item['headline'].lower()
                sentence = sentence.replace(",", " , ")
                sentence = sentence.replace(".", " . ")
                sentence = sentence.replace("-", " - ")
                sentence = sentence.replace("/", " / ")
                soup = BeautifulSoup(sentence)
                sentence = soup.get_text()
                words = sentence.split()
                filtered_sentence = ""
                for word in words:
                    word = word.translate(table)
                    if word not in stopwords:
                        filtered_sentence = filtered_sentence + word + " "
                all_sentences.append(filtered_sentence)
                all_labels.append(item['is_sarcastic'])
                all_urls.append(item['article_link'])

# Split data into training and test sets
training_size = 23000
training_sentences = all_sentences[:training_size]
testing_sentences = all_sentences[training_size:]
training_labels = all_labels[:training_size]
testing_labels = all_labels[training_size:]

# Tokenize the text
vocab_size = 20000
max_length = 10
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(word_index)
