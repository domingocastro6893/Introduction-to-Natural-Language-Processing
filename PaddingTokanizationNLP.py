from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the sentences
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences
padded = pad_sequences(sequences)

# Print the padded sequences
print("Padded Sequences:")
print(padded)

# Postpad the sequences
padded_post = pad_sequences(sequences, padding='post')
print("\nPadded Sequences (Postpadding):")
print(padded_post)

# Limit the maximum length of sequences
padded_maxlen = pad_sequences(sequences, maxlen=6)
print("\nPadded Sequences (Maxlen):")
print(padded_maxlen)

# Truncate longer sequences from the end
padded_truncate = pad_sequences(sequences, maxlen=6, truncating='post')
print("\nPadded Sequences (Truncating):")
print(padded_truncate)
