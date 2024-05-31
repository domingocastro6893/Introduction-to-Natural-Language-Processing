import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import string

# Define a list of stopwords
stopwords = ["a", "about", "above", "an", "and", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"]

# Create a translation table to remove punctuation
table = str.maketrans('', '', string.punctuation)

# Initialize a list to store preprocessed sentences
imdb_sentences = []

# Load the IMDb dataset from TensorFlow Datasets
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))

# Iterate through the dataset, preprocess the text, and append to the list
for item in train_data:
    # Decode and convert text to lowercase
    sentence = str(item['text'].decode('UTF-8')).lower()

    # Remove HTML tags
    soup = BeautifulSoup(sentence, 'html.parser')
    sentence = soup.get_text()

    # Remove punctuation and stopwords
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence += word + " "

    imdb_sentences.append(filtered_sentence)

# Tokenize the preprocessed sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

# Print the word index
print("Word Index:")
print(tokenizer.word_index)

# Example of decoding a sequence
decoded_review = ' '.join([tokenizer.index_word.get(i, '?') for i in sequences[0]])
print("\nDecoded Review:")
print(decoded_review)
