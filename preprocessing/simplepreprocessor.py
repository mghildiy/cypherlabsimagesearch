import cv2


class SimplePreprocessor:

    def __init__(self, width: object, height: object, inter: object = cv2.INTER_AREA) -> object:
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)
