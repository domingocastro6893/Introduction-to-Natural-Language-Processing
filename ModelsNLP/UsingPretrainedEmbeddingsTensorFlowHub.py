import tensorflow as tf
import tensorflow_hub as hub
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

# Download pretrained embedding layer from TensorFlow Hub
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
    output_shape=[20], input_shape=[], dtype=tf.string, trainable=False
)

# Create the model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(training_sentences, training_labels, epochs=50, validation_data=(testing_sentences, testing_labels), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(testing_sentences, testing_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
