import os
import cv2
import pandas as pd
from imutils import paths

imagePaths = list(paths.list_images('C:\work\ml\work\cypherlabsimagesearch\data\dogbreed\images'))


def csvToDictionary(pathToCsv):
    df = pd.read_csv(pathToCsv, index_col=0)
    d = df['breed'].to_dict()
    return d

imageIdToBreed = csvToDictionary('C:\work\ml\work\cypherlabsimagesearch\data\dogbreed\labels.csv')

targetCount = 0;
for (i, imagePath) in enumerate(imagePaths):
    imageName = imagePath.split(os.path.sep)[-1]
    imageId = imageName.split(".")[0]

    breed = imageIdToBreed[imageId]
    print('Breed name is '+breed)
    if(breed != 'english_foxhound'):
        continue

    targetCount += 1

    orgImage = cv2.imread(imagePath)
    #cv2.imshow("Image", orgImage)
    #cv2.waitKey(0)

print("Total count for target breed is:"+ str(targetCount))