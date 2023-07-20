import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class CNNTraining:
    def __init__(self, image_path):
        self.image_path = image_path

    def preprocess_image(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        dilation = cv2.dilate(thresh, (3, 3), iterations=1)
        eroded = cv2.erode(dilation, (3, 3), iterations=2)
        return eroded

    def split_image_into_cells(self, image):
        cells = [np.hsplit(row, 56) for row in np.vsplit(image, 8)]
        x = np.array(cells)
        return x

    def prepare_training_data(self, cells):
        train = cells[:, :].reshape(-1, 50, 50, 1).astype(np.float32)
        
        train_labels = np.repeat([0, 1, 2, 3], 112)
        
        return train, train_labels

    def create_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def save_training_data(self, train, train_labels, save_path):
        self.cnn_model = self.create_cnn_model()
        self.cnn_model.fit(train, train_labels, epochs=10, batch_size=32)
        self.cnn_model.save(save_path)

    def save_cell_images(self, cells, save_path):
        cv2.imwrite(save_path + 'letters_2t_E.png', cells[0, 22])
        cv2.imwrite(save_path + 'letters_2t_W.png', cells[2, 22])
        cv2.imwrite(save_path + 'letters_2t_S.png', cells[4, 22])
        cv2.imwrite(save_path + 'letters_2t_N.png', cells[6, 22])

    def run_training(self):
        eroded_image = self.preprocess_image()
        cells_data = self.split_image_into_cells(eroded_image)
        train_data, train_labels = self.prepare_training_data(cells_data)

        self.save_training_data(train_data, train_labels, '/home/jisu/angchicken_ros2_ws/src/text_detection/cnn_letters/my_model.keras')
        self.save_cell_images(cells_data, '/home/jisu/angchicken_ros2_ws/src/text_detection/cnn_letters/')


if __name__ == "__main__":
    image_path = "/home/jisu/angchicken_ros2_ws/src/text_detection/cnn_letters/notNIST_3t.png"

    cnn_training = CNNTraining(image_path)
    cnn_training.run_training()
