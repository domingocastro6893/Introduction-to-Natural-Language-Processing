import tensorflow_datasets as tfds


(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)

encoder = info.features['text'].encoder
print('Vocabulary size:', encoder.vocab_size)
print('List of subwords:', encoder.subwords)

sample_string = 'Today is a sunny day'
encoded_string = encoder.encode(sample_string)
print('Encoded string:', encoded_string)

original_string = encoder.decode(encoded_string)
print('Decoded string:', original_string)
