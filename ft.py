import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

# Load a larger dataset (here, the English part of the Tatoeba dataset)
df = pd.read_csv('https://object.pouta.csc.fi/Tatoeba-Challenge-2017/eng.csv')
sentences = df['eng'].astype(str).tolist()

# Define the tokenizer (converts words to integers)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Convert the sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to a fixed length
max_length = 10
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

# Convert the padded sequences to one-hot vectors
one_hot_sequences = to_categorical(padded_sequences, num_classes=len(tokenizer.word_index)+1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(one_hot_sequences[:,:-1], one_hot_sequences[:,-1], test_size=0.2)

# Define the model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length-1),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the English dataset
model.fit(X_train, y_train, epochs=10)

# Load a smaller dataset in a different language (here, Spanish)
df = pd.read_csv('https://object.pouta.csc.fi/Tatoeba-Challenge-2017/spa.csv')
sentences = df['spa'].astype(str).tolist()

# Convert the sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to a fixed length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

# Convert the padded sequences to one-hot vectors
one_hot_sequences = to_categorical(padded_sequences, num_classes=len(tokenizer.word_index)+1)

# Fine-tune the model on the Spanish dataset
model.fit(one_hot_sequences[:,:-1], one_hot_sequences[:,-1], epochs=10)

# Use the model to predict the next word in a sentence in Spanish
test_sentence = 'Me gusta'
test_sequence = tokenizer.texts_to_sequences([test_sentence])
test_padded_sequence = pad_sequences(test_sequence, maxlen=max_length-1, padding='pre')
prediction = model.predict(test_padded_sequence)

# Convert the predicted integer back to a word
predicted_word_index = tf.argmax(prediction, axis=1).numpy()[0]
predicted_word = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(predicted_word_index)]

# Print the predicted word
print(predicted_word)
