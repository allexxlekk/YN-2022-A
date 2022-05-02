import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense
from keras.optimizers import gradient_descent_v2
from sklearn.model_selection import KFold
import deliciousMIL_Loader as loader
from data_prep import center_dataset
import matplotlib.pyplot as plt

H = 0.05  #  Learning rate.
M = 0.6  # Momentum.
λ = 0.1  # Regularization parameter.
N_INPUTS = 8520  #  Same as number of features (words).
N_OUTPUTS = 20  #  Same as number of labels.
EPOCHS = 30
BATCH_SIZE = 15

# source Keras.io
def plot_result(item):
    plt.plot(best_history.history[item], label=item)
    plt.plot(best_history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


# Early stopping based on validation accuracy.
es = keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy", patience=5, mode="max", min_delta=0.001
)


def create_model(n_input, n_outputs, loss_f):
    model = Sequential()
    model.add(
        Dense(
            8520,
            input_dim=n_input,
            activation="relu",
        )
    )  #  First hidden layer.
    model.add(
        Dense(500, activation="relu", kernel_regularizer=regularizers.l2(λ))
    )  # Second hidden layer.
    model.add(Dense(n_outputs, activation="sigmoid"))  # Output layer.
    model.compile(
        loss=loss_f,
        optimizer=gradient_descent_v2.SGD(learning_rate=H, momentum=M),
        metrics=["binary_accuracy", "mse"],
    )
    return model


print("Loading Training Data...")
train_data, train_labels = loader.get_dataset("train.dat")

print("Loading Validation Data...")
val_data, val_labels = loader.get_dataset("test.dat")

train_data = center_dataset(train_data)
val_data = center_dataset(val_data)

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)

model_results = []
best_accuracy = 0

for i, (train, test) in enumerate(kfold.split(train_data)):

    X_train, X_test = train_data[train], train_data[test]
    Y_train, Y_test = train_labels[train], train_labels[test]

    model = create_model(N_INPUTS, N_OUTPUTS, "binary_crossentropy")
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[es],
    )

    _, categorical_acc, x = model.evaluate(X_test, Y_test)
    model_results.append(categorical_acc)

    # Keep the best performing network
    if categorical_acc > best_accuracy:
        index = i + 1
        best_model = model
        best_history = history
        best_accuracy = categorical_acc

# Print results from 5 fold CV
for i, result in enumerate(model_results):
    print(f"Fold:{i+1} Αccuracy on the test set: {round(result * 100, 2)}%.")

# Print results from the best performing network.
train_CE, train_acc, train_MSE = best_model.evaluate(
    train_data, train_labels, verbose=0
)
test_CE, test_acc, test_MSE = best_model.evaluate(val_data, val_labels, verbose=0)

print(f"Model number {index} chosen")
print(
    f"Train: {round(train_acc * 100, 2)}%, CE loss: {train_CE} ,MSE loss: {train_MSE}"
)
print(f"Test: {round(test_acc * 100, 2)}%, CE loss: {test_CE} ,MSE loss: {test_MSE}")

plot_result("loss")
plot_result("binary_accuracy")
