import cv2
import uuid
import os
import numpy as np

def rotate(image, angle=90, scale=1.0):
    '''
    Rotate the image
    :param image: image to be processed
    :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    :param scale: Isotropic scale factor.
    '''
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image

def flip(image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

source_dir = r'C:\work\ml\work\cypherlabsimagesearch\nn\conv\dogbreeddataaugmenter\original_data'
save_dir = r'C:\work\ml\work\cypherlabsimagesearch\nn\conv\dogbreeddataaugmenter\augmeneted_data'

for filename in os.listdir(source_dir):
    image = cv2.imread(os.path.join(source_dir, filename))
    img = image.copy()
    #img_flip = self.flip(img, vflip=True, hflip=False)
    for angle in np.arange(0, 360, 90):
        img_rot = rotate(img, angle=-angle)
        #cv2.imwrite(save_dir+'%s' %str(uuid.uuid1())+'.jpg', img_rot)
        img_flip_vertical = flip(img_rot, vflip=True, hflip=False)
        img_flip_horizontal = flip(img_rot, vflip=False, hflip=True)
        cv2.imwrite(os.path.join(save_dir, str(uuid.uuid1())+'_rot.jpg'), img_rot)
        cv2.imwrite(os.path.join(save_dir, str(uuid.uuid1()) + '_vertical_flip.jpg'), img_flip_vertical)
        cv2.imwrite(os.path.join(save_dir, str(uuid.uuid1()) + '_horizontal_flip.jpg'), img_flip_horizontal)

#img_gaussian = add_GaussianNoise(img)

#name_int = self.name[:len(self.name) - 4]
#cv2.imwrite(save_path + '%s' % str(name_int) + '_vflip.jpg', img_flip)
#cv2.imwrite(save_dir + uuid.uuid1() +'_rot.jpg', img_rot)
#cv2.imwrite(save_path + '%s' % str(name_int) + '_GaussianNoise.jpg', img_gaussian)