import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the dataset (list of sentences)
sentences = ['The cat sat on the', 'I like to eat', 'She went to the']

# Define the tokenizer (converts words to integers)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Convert the sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to a fixed length
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

# Define the model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, padded_sequences, epochs=50)

# Use the model to predict the next word in a sentence
test_sentence = 'I like to eat'
test_sequence = tokenizer.texts_to_sequences([test_sentence])
test_padded_sequence = pad_sequences(test_sequence, maxlen=max_length, padding='pre')
prediction = model.predict(test_padded_sequence)

# Convert the predicted integer back to a word
predicted_word_index = tf.argmax(prediction, axis=1).numpy()[0]
predicted_word = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(predicted_word_index)]

# Print the predicted word
print(predicted_word)
