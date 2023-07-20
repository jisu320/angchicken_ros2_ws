import cv2
import numpy as np

class KNNLetterDetection:
    def __init__(self, file_name):
        self.file_name = file_name
        self.cap = cv2.VideoCapture(4,cv2.CAP_V4L2)
        self.train, self.train_labels = self.load_train_data()
        self.knn = cv2.ml.KNearest_create()

    def img_cvt(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        dilation = cv2.dilate(thresh, (3, 3), iterations=2)
        eroded = cv2.erode(dilation, (3, 3), iterations=2)
        return eroded

    def find_contour(self, origin, img):
        origin_cp = origin.copy()
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, 3)
        cv2.drawContours(origin_cp, contours, -1, (0, 255, 0), 3)
        img_crop = []

        for i in range(len(contours)):
            cv2.putText(origin_cp, str(i), tuple(contours[i][0])[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(contours[i])
            crop = img[y:y+h, x:x+w]
            img_crop.append(crop)

        return img_crop, origin_cp

    def load_train_data(self):
        with np.load(self.file_name) as data:
            train = data['train']
            train_labels = data['train_labels']
        return train, train_labels

    def cvt_img20(self, img):
        gray_resize = cv2.resize(img, (50, 50))
        return gray_resize.reshape(-1, 2500).astype(np.float32)

    def check(self, resized_img):
        self.knn.train(self.train, cv2.ml.ROW_SAMPLE, self.train_labels)
        ret, result, neighbours, dist = self.knn.findNearest(resized_img, k=3)

        dist_avg = np.float32(int(sum(dist[0])) / int(len(dist[0])))
        if dist_avg < 20000000:
            if result == 0:
                print('E is detected')
            elif result == 1:
                print('W is detected')
            elif result == 2:
                print('S is detected')
            elif result == 3:
                print('N is detected')

            return result
        else:
            pass

    def run(self):
        while True:
            ret, img = self.cap.read()

            if not ret:
                print("fail")
                exit()
            else:
                cvted_img = self.img_cvt(img)
                img_crop, origin_cp = self.find_contour(img, cvted_img)
                cvted_img20 = []

                for i in range(len(img_crop)):
                    cvted_img20.append(self.cvt_img20(img_crop[i]))

                if cvted_img20:
                    for i in range(len(cvted_img20)):
                        checking = self.check(cvted_img20[i])
                        if checking is not None:
                            print(checking)

                #cv2.imshow('cvted_img20', cvted_img20[0].reshape(50, 50))
                #v2.imshow('origin', img)
                cv2.imshow('origin_cp', origin_cp)

                key = cv2.waitKey(500)
                if key == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    FILE_NAME = '/home/jisu/angchicken_ros2_ws/src/text_detection/knn_letters/trained_letters_2t.npz'
    knn_letter_detection = KNNLetterDetection(FILE_NAME)
    knn_letter_detection.run()