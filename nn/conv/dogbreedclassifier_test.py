import numpy as np
from imutils import paths
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
import cv2
import os
import pandas as pd

#classLabels = ['afghan_hound', 'basenji', 'bernese_mountain_dog', 'entlebucher',
# 'great_pyrenees', 'maltese_dog', 'pomeranian', 'samoyed',
# 'scottish_deerhound', 'shih-tzu']

#classLabels = ['brabancon_griffon', 'briard', 'eskimo_dog', 'german_shepherd',
# 'giant_schnauzer', 'golden_retriever', 'komondor', 'otterhound',
# 'tibetan_mastiff', 'walker_hound']

classLabels = ['affenpinscher','afghan_hound','african_hunting_dog','airedale',
 'american_staffordshire_terrier','appenzeller','australian_terrier',
 'basenji','basset','beagle','bedlington_terrier','bernese_mountain_dog',
 'black-and-tan_coonhound','blenheim_spaniel','bloodhound','bluetick',
 'border_collie','border_terrier','borzoi','boston_bull',
 'bouvier_des_flandres','boxer','brabancon_griffon','briard',
 'brittany_spaniel','bull_mastiff','cairn','cardigan',
 'chesapeake_bay_retriever','chihua','hua','chowclumber','cocker_spaniel',
 'collie','curly-coated_retriever','dandie_dinmont','dhole','dingo',
 'doberman','english_foxhound','english_setter','english_springer',
 'entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog',
 'german_shepherd','german_short-haired_pointer','giant_schnauzer',
 'golden_retriever','gordon_setter','great_dane','great_pyrenees',
 'greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter',
 'irish_terrier','irish_water_spaniel','irish_wolfhound',
 'italian_greyhound','japanese_spaniel','keeshond','kelpie',
 'kerry_blue_terrier','komondor','kuvasz','labrador_retriever',
 'lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog',
 'mexican_hairless','miniature_pinscher','miniature_poodle',
 'miniature_schnauzer','newfoundland','norfolk_terrier',
 'norwegian_elkhound','norwich_terrier','old_english_sheepdog',
 'otterhound','papillon','pekinese','pembroke','pomeranian','pug',
 'redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki',
 'samoyed','schipperke','scotch_terrier','scottish_deerhound',
 'sealyham_terrier','shetland_sheepdog','shih-tzu','siberian_husky',
 'silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier',
 'standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff',
 'tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound',
 'weimaraner','welsh_springer_spaniel','west_highland_white_terrier',
 'whippet','wire-haired_fox_terrier','yorkshire_terrier']


#classLabels = ['boston_bull','bouvier_des_flandres', 'rhodesian_ridgeback', 'bluetick']
#classLabels = ['boston_bull', 'rhodesian_ridgeback', 'bluetick']
#classLabels = ['brabancon_griffon', 'golden_retriever', 'komondor']
#classLabels = ['boston_bull','bouvier_des_flandres', 'collie', 'dandie_dinmont', 'rhodesian_ridgeback']
#classLabels = ['bluetick', 'boston_bull','bouvier_des_flandres', 'rhodesian_ridgeback']
#classLabels = ['african_hunting_dog', 'bluetick', 'boston_bull', 'bouvier_des_flandres',
# 'english_foxhound', 'kelpie', 'rhodesian_ridgeback', 'schipperke']
classLabels = ['australian_terrier', 'blenheim_spaniel', 'irish_wolfhound','lakeland_terrier', 'miniature_pinscher', 'saluki']

imagePaths = np.array(list(paths.list_images('../../data/dogbreed/test_images/irish_wolfhound')))
print(imagePaths)
#idxs = np.random.randint(0, len(imagePaths), size=(10,))
#imagePaths = imagePaths[idxs]
#print(imagePaths)

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
testData = sdl.loadImages(imagePaths)
testData = testData.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model("C:\\work\\ml\\work\\cypherlabsimagesearch\\nn\\conv\\saved_models\\"
    "dogbreeedclassifier_100&&australian_terrier&&blenheim_spaniel&&irish_wolfhound&&lakeland_terrier&&miniature_pinscher&&saluki_weights.hdf5")

print("[INFO] predicting...")
probVectors = model.predict(testData, batch_size=32)
preds = probVectors.argmax(axis=1)

imageIds = []

df = pd.DataFrame(probVectors, columns = classLabels)

for (i, imagePath) in enumerate(imagePaths):
    imageName = imagePath.split(os.path.sep)[-1]
    imageId = imageName.split(".")[0]
    imageIds.append(imageId)
    print(probVectors[i][preds[i]])
    # load the example image, draw the prediction, and display it to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


#df.insert(0, 'id', imageIds)
#df.to_excel("C:\work\ml\work\cypherlabsimagesearch\data\dogbreed\output.xlsx")