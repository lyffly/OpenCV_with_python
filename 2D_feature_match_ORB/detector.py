import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_image', type=str)
parser.add_argument('--query_image', type=str)
args = parser.parse_args()

if args.train_image is None:
    exit(1)

class pipeline:
    def __init__(self, train_image):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kp1 = self.orb.detect(train_image, None)
        self.kp1, self.des1 = self.orb.compute(train_image, self.kp1)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
        self.train_image = train_image




    def preprocess(self, img):
        bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2 = self.orb.detect(img, None)
        kp2, des2 = self.orb.compute(bw_image, kp2)
        # matches = self.bf.match(self.des1, des2)
        if des2 is not None and len(self.des1) > 0 and len(des2) > 0:
            matches = self.flann.knnMatch(np.float32(self.des1), np.float32(des2), k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.90 * n.distance:
                    good.append(m)
            print(len(good))

            # src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            for pt in dst_pts:
                img = cv2.circle(img, (pt[0][0], pt[0][1]), 10, (0, 255, 0), -1)
                cv2.putText(img,"dst",(pt[0][0], pt[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))
            return img, dst_pts
        else:
            return img, []

    def sliding_window(self, img):
        stepSize = 1000
        (w_width, w_height) = (self.train_image.shape[0], self.train_image.shape[1])  # window size
        best_point = None
        num_best_points = 0
        for x in range(0, img.shape[1] - w_width, stepSize):
            for y in range(0, img.shape[0] - w_height, stepSize):
                window = img[x:x + w_width, y:y + w_height, :]
                _, points = self.preprocess(window)
                if len(points) > num_best_points:
                    best_point = (x + w_width, y + w_height)
                    num_best_points = len(points)

        # TODO: Write a better algo here to determine a selection of points we can use to consider it as the object we want
        if best_point is not None:
            img = cv2.circle(img, (best_point[0], best_point[1]), 30, (0, 0, 255), -1)
        return img

    def visualisation(self, img):
        img = self.sliding_window(img)
        return img


# main
if __name__ == '__main__':
    cap = None
    if args.query_image is None:
        cap = cv2.VideoCapture(2)

    train_image = cv2.imread(args.train_image)  # trainImage
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    # train_image = cv2.resize(train_image, (0, 0), fx=0.3, fy=0.3)  # Scale resizing

    my_pipeline = pipeline(train_image)

    while(True):
        if args.query_image is None:
            ret, frame = cap.read()
        else:
            frame = cv2.imread(args.query_image)

        # initialize
        frame_size = frame.shape
        frame_width = frame_size[1]
        frame_height = frame_size[0]

        # frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)  # Scale resizing

        thresholds = {}

        visualisation = my_pipeline.visualisation(frame)

        # numpy_horizontal_concat = np.concatenate((frame, visualisation), 1)

        #cv2.imshow('image', cv2.resize(visualisation, (0, 0), fx=0.3, fy=0.3))
        cv2.imshow('image', visualisation)

        cv2.waitKey(1)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


