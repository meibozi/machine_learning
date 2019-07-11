import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D, ZeroPadding2D


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image)
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx])
        title = str(i) + "," + label_dict[labels[idx][0]]
        if len(prediction) > 0:
            title += "=>" + label_dict[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):


label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
# print(len(x_train_image))
# print(len(x_test_image))
#
# print(x_train_image.shape)
# print(y_train_label.shape)
#
# plot_image(x_train_image[0])
# print(y_train_label[0])
#
# plot_images_labels_prediction(x_test_image,y_test_label,[],idx=340)


x_Train_normalize = x_train_image.astype("float32") / 255.0
x_Test_normalize = x_test_image.astype("float32") / 255.0

print(y_train_label[:5])

y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
print(y_TrainOneHot.shape)

print(y_TrainOneHot[:5])

model = Sequential()
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        input_shape=(32, 32, 3),
        activation="relu",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units=1024, kernel_initializer="normal", activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=10, kernel_initializer="normal", activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

try:
    model.load_weights("cifarCnnModel.h5")
    print("continue training")
except:
    print("new training")

train_history = model.fit(
    x=x_Train_normalize,
    y=y_TrainOneHot,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=1,
)
scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print(scores)

prediction = model.predict_classes(x_Test_normalize)
# plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340)

y_test_label_reshape = y_test_label.reshape(-1)
a = pd.crosstab(
    y_test_label_reshape, prediction, rownames=["label"], colnames=["predict"]
)
print(a)

df = pd.DataFrame({"label": y_test_label_reshape, "predict": prediction})
b = df[(df.label == 5) & (df.predict == 3)]
print(b)

model.save_weights("cifarCnnModel.h5")
