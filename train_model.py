# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Flatten, Dense
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Load your dataset
# df = pd.read_csv('./IMDB_dataset.csv')

# # Preprocess the data
# tokenizer = Tokenizer(oov_token='<OOV>')
# tokenizer.fit_on_texts(df['review'])
# sequences = tokenizer.texts_to_sequences(df['review'])
# padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment'], test_size=0.2, random_state=42)

# # Build and train the model
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=100))
# model.add(Flatten())
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# # Save the model
# model.save('trained_model.h5')

import tensorflow as tf
print(tf.__version__)