import cv2
import numpy as np

class KNNTraining:
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
        train = cells[:, :].reshape(-1, 2500).astype(np.float32)
        
        train_labels = np.repeat([0, 1, 2, 3], 112)[:, np.newaxis]
        
        return train, train_labels

    def save_training_data(self, train, train_labels, save_path):
        np.savez(save_path, train=train, train_labels=train_labels)

    def save_cell_images(self, cells, save_path):
        cv2.imwrite(save_path + 'letters_2t_E.png', cells[0, 22])
        cv2.imwrite(save_path + 'letters_2t_W.png', cells[2, 22])
        cv2.imwrite(save_path + 'letters_2t_S.png', cells[4, 22])
        cv2.imwrite(save_path + 'letters_2t_N.png', cells[6, 22])


if __name__ == "__main__":
    image_path = "/home/jisu/angchicken_ros2_ws/src/text_detection/knn_letters/notNIST_3t.png"
    save_path = "/home/jisu/angchicken_ros2_ws/src/text_detection/knn_letters/trained_letters_2t.npz"

    knn_training = KNNTraining(image_path)
    eroded_image = knn_training.preprocess_image()
    cells_data = knn_training.split_image_into_cells(eroded_image)
    train_data, train_labels = knn_training.prepare_training_data(cells_data)
    knn_training.save_training_data(train_data, train_labels, save_path)
    knn_training.save_cell_images(cells_data, '/home/jisu/angchicken_ros2_ws/src/text_detection/knn_letters/')
