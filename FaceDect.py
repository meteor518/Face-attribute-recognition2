import dlib
import cv2
import numpy as np

dlib_model = './pre_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model)


def face_dect(img, padding_ratio1=0.12):
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        x = d.left()
        y = d.top()
        w = d.right() - x
        h = d.bottom() - y
        img = img[max(0, int(y - h * padding_ratio1)):int(y + h * (1 + padding_ratio1)),
              max(0, int(x - w * padding_ratio1)):int(x + w * (1 + padding_ratio1))]

        return img


def feature_68(img):
    face_landmark = np.zeros((68, 2))
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
        # shape = predictor(img, rec)
        shape = predictor(img, d)
        #  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
        for i in range(68):
            face_landmark[i, 0] = shape.part(i).x
            face_landmark[i, 1] = shape.part(i).y
            #         print(x, y)
            # cv2.circle(img, (x, y), 1, (0, 0, 255))

    return face_landmark