import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from utils.utils import printFrequency, plotFrequencyBar

class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            # we assume image path has format /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            cv2.imshow("Image", image)
            cv2.waitKey(0)

            # if preprocessors are available, we apply them consecutively on image
            if self.preprocessors is not None:
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return tuple of data and labels
        return (np.array(data), np.array(labels))

    def getBreedToNumAngIncrments(self, breedToFrequency, finalCount):
        breedToAngleIncr = {}
        for breed in breedToFrequency:
            #print(breed)
            #print(breedToFrequency[breed])
            totalExtraImagesNeeded = finalCount - breedToFrequency[breed]
            extrasPerImage = math.ceil(totalExtraImagesNeeded / breedToFrequency[breed])
            numIncr = math.ceil(extrasPerImage / 8)
            breedToAngleIncr[breed] = numIncr

        return breedToAngleIncr


    def loadfromlabels(self, imagePaths, labelPath, verbose=-1):
        #dogbreeds = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher',
        #'bernese_mountain_dog', 'shih-tzu', 'great_pyrenees', 'pomeranian',
        #'basenji', 'samoyed']
        #dogbreeds = ['walker_hound', 'german_shepherd', 'giant_schnauzer', 'tibetan_mastiff',
        #'otterhound', 'brabancon_griffon', 'golden_retriever', 'komondor',
        #'briard', 'eskimo_dog']
        #dogbreeds = ['boston_bull', 'rhodesian_ridgeback', 'bluetick']
        #dogbreeds = ['brabancon_griffon', 'golden_retriever', 'komondor']
        #dogbreeds = ['boston_bull','bluetick','african_hunting_dog','schipperke','rhodesian_ridgeback',
        #            'kelpie','english_foxhound','bouvier_des_flandres']
        dogbreeds = ['lakeland_terrier', 'saluki', 'irish_wolfhound','blenheim_spaniel', 'miniature_pinscher','australian_terrier']


        data = []
        labels = []

        dict, breedToFrequency = self.csvToDictionary(labelPath)

        breedToNumAngIncrments = self.getBreedToNumAngIncrments(breedToFrequency, 3000)


        for (i, imagePath) in enumerate(imagePaths):
            imageName = imagePath.split(os.path.sep)[-1]
            imageId = imageName.split(".")[0]


            if (dict[imageId] not in dogbreeds):
                continue

            #print(dict[imageId])
            augmentedImages = []

            orgImage = cv2.imread(imagePath)
            #cv2.imshow("Image", orgImage)
            #cv2.waitKey(0)
            augmentedImages.append(orgImage)

            numIncrements = breedToNumAngIncrments[dict[imageId]]

            #############
            maxAngleRot = 5 + (numIncrements-1) * 5 + 1
            for angle in np.arange(5, maxAngleRot, 5):
                img_rot_cw = self.rotate(orgImage, angle=-angle)
                #cv2.imshow("Image", img_rot_cw)
                #cv2.waitKey(0)
                augmentedImages.append(img_rot_cw)
                img_flip_horizontal = self.flip(img_rot_cw, vflip=False, hflip=True)
                #cv2.imshow("Image", img_flip_horizontal)
                #cv2.waitKey(0)
                augmentedImages.append(img_flip_horizontal)

                img_rot_ccw = self.rotate(orgImage, angle=angle)
                #cv2.imshow("Image", img_rot_ccw)
                #cv2.waitKey(0)
                augmentedImages.append(img_rot_ccw)
                img_flip_horizontal = self.flip(img_rot_ccw, vflip=False, hflip=True)
                #cv2.imshow("Image", img_flip_horizontal)
                #cv2.waitKey(0)
                augmentedImages.append(img_flip_horizontal)

                img_translate = self.translate(orgImage,random.randint(-50, 50), random.randint(-50, 50))
                #cv2.imshow("Image", img_translate)
                #cv2.waitKey(0)
                augmentedImages.append(img_translate)
                img_translate = self.translate(orgImage, random.randint(-50, 50), random.randint(-50, 50))
                #cv2.imshow("Image", img_translate)
                #cv2.waitKey(0)
                augmentedImages.append(img_translate)
                img_translate = self.translate(orgImage, random.randint(-50, 50), random.randint(-50, 50))
                #cv2.imshow("Image", img_translate)
                #cv2.waitKey(0)
                augmentedImages.append(img_translate)
                img_translate = self.translate(orgImage, random.randint(-50, 50), random.randint(-50, 50))
                #cv2.imshow("Image", img_translate)
                #cv2.waitKey(0)
                augmentedImages.append(img_translate)
            #############

            for image in augmentedImages:
                if self.preprocessors is not None:
                    for preprocessor in self.preprocessors:
                        image = preprocessor.preprocess(image)
                        #cv2.imshow("Image", image)
                        #cv2.waitKey(0)
                data.append(image)
                labels.append(dict[imageId])


            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))


        #plotFrequencyBar(printFrequency(labels))
        printFrequency(labels)

        return (np.array(data), np.array(labels))



    def csvToDictionary(self, pathToCsv):
        df = pd.read_csv(pathToCsv, index_col=0)
        #fig, ax = plt.subplots()
        #df['breed'].value_counts().plot(ax=ax, kind='bar')
        #plt.show()
        d = df['breed'].to_dict()
        return (d, df['breed'].value_counts().to_dict())

    def rotate(self, image, angle=90, scale=1.0):
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def translate(self, image, x, y):
        height, width = image.shape[:2]
        T = np.float32([[1, 0, x], [0, 1, y]])
        translatedImage = cv2.warpAffine(image, T, (width, height))

        return translatedImage

    def loadImages(self, imagePaths):
        images = []
        for (i, imagePath) in enumerate(imagePaths):
            imageName = imagePath.split(os.path.sep)[-1]
            imageId = imageName.split(".")[0]

            image = cv2.imread(imagePath)

            if self.preprocessors is not None:
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)

            images.append(image)

        return np.array(images)






