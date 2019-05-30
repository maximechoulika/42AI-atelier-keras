""" Helper file """

def basic_model(input_shape, nb_classes):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import Conv2D
    from keras.layers import MaxPool2D

    model=Sequential()

    model.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation="relu",input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=16,kernel_size=(4,4),padding="Same",activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=32,kernel_size=(4,4),padding="Same",activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,activation="softmax"))

    return model

def compile_model(model):
    from keras.optimizers import Adam
    optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])

    return model

def load_dir(path):
    import os
    from glob import glob

    path = os.path.abspath(path)
    dirs = glob(path + "/*")
    data = {}
    for dir in dirs:
        class_name = dir.split('/')[-1].replace(' ', '_').lower()
        data[class_name] = glob(dir + '/*')

    return data

def load_fruit_dataset(path, shuffle=False):
    import numpy as np
    train_path = "{}/Training".format(path)
    test_path  = "{}/Test".format(path)

    train_set  = load_dir(train_path)
    test_set   = load_dir(test_path)

    class_idx = {}
    for i, key in enumerate(train_set.keys()):
        class_idx[key] = i
    # sanitize data
    train = []
    test  = []
    for key, val in class_idx.items():
        for train_img, test_img in zip(train_set[key], test_set[key]):
            train.append([val, train_img])
            test.append([val, test_img])
        del train_set[key]
        del test_set[key]

    train = np.array(train)
    test  = np.array(test)
    if shuffle:
        np.random.shuffle(train)
        np.random.shuffle(test)

    return train, test, class_idx

def load_image(path):
    import cv2
    img = cv2.imread(path)

    return img

def show_image(img, size=None):
    import cv2
    import numpy as np
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('image',np.uint8(img))
    cv2.waitKey(0)


if __name__ == "__main__":
    train_set, test_set, class_idx = load_fruit_dataset('dataset', shuffle=True)
    print(test_set[:10])
