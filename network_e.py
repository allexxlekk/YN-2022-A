import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from deliciousMIL_Loader import get_dataset_E
import matplotlib.pyplot as plt
from keras.optimizers import gradient_descent_v2

H = 0.001  #  Learning rate.
EPOCHS = 30
BATCH_SIZE = 15

# source Keras.io
def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


es = keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy", patience=5, mode="max", min_delta=0.001
)
raw_train_data, train_labels, train_padding = get_dataset_E("train.dat")
raw_test_data, test_labels, test_padding = get_dataset_E("test.dat")

max_padding = train_padding if train_padding > test_padding else test_padding

# integer encode the documents
vocab_size = 8550
encoded_train_data = [one_hot(d, vocab_size) for d in raw_train_data]
encoded_test_data = [one_hot(d, vocab_size) for d in raw_test_data]

# pad documents to a max length of 4 words
train_data = pad_sequences(encoded_train_data, maxlen=max_padding, padding="post")
test_data = pad_sequences(encoded_test_data, maxlen=max_padding, padding="post")

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 70, input_length=max_padding))
model.add(Flatten())
model.add(Dense(8520, activation="relu"))
model.add(Dense(500, activation="relu"))
model.add(Dense(20, activation="sigmoid"))

model.compile(
    optimizer=gradient_descent_v2.SGD(learning_rate=H, momentum=0.7),
    loss="binary_crossentropy",
    metrics=["binary_accuracy", "mse"],
)

history = model.fit(
    train_data,
    train_labels,
    epochs=50,
    validation_data=(test_data, test_labels),
    batch_size=BATCH_SIZE,
    callbacks=[es],
)

loss, accuracy, mse = model.evaluate(test_data, test_labels)
print(f"Accuracy: {round(accuracy * 100, 2)}%")

plot_result("loss")
plot_result("binary_accuracy")
