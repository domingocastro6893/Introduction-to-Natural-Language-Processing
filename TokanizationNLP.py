from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Define the sentences
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

# Step 2: Initialize the Tokenizer with OOV token
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Step 3: Fit the tokenizer on the sentences
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Print the word index to see the token mapping
print("Word Index:", word_index)

# Step 4: Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)
print("Sequences:", sequences)

# Step 5: Define new sentences (test data)
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

# Step 6: Convert test sentences to sequences
test_sequences = tokenizer.texts_to_sequences(test_data)
print("Test Sequences:", test_sequences)

# Optional: Decode the sequences back to words to verify
def decode_sequence(sequence, word_index):
    reverse_word_index = {value: key for key, value in word_index.items()}
    return [reverse_word_index.get(num, "<OOV>") for num in sequence]

# Decode and print the original sentences from sequences
decoded_sentences = [decode_sequence(seq, word_index) for seq in sequences]
print("Decoded Sentences:", decoded_sentences)

decoded_test_sentences = [decode_sequence(seq, word_index) for seq in test_sequences]
print("Decoded Test Sentences:", decoded_test_sentences)
