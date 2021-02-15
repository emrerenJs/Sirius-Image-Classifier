import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


class ImagePreProcessor:
    def __init__(self,workdir,imagedata):
        self.workdir = workdir
        self.imagedata = imagedata

    def image_augmentation(self,augmentation_count = 10):
        path = self.workdir
        augmentated_path = os.path.join(path, "augmentated_images")
        if not os.path.exists(augmentated_path):
            os.mkdir(augmentated_path)
            workdir = os.path.join(path, "images")

            labels = os.listdir(workdir)
            for label in labels:
                images_path = os.listdir(os.path.join(workdir, label))
                os.mkdir(os.path.join(augmentated_path, label))
                for image_path in images_path:
                    try:
                        img_path = os.path.join(workdir, label, image_path)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        datagen = ImageDataGenerator(
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode="nearest"
                        )
                        x = img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        i = 0
                        img_format, img_name = image_path[::-1].split(".", 1)
                        img_name = img_name[::-1]
                        img_format = img_format[::-1]
                        for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(augmentated_path, label),
                                                  save_prefix=img_name, save_format=img_format):
                            i += 1
                            if i > augmentation_count - 1:
                                break
                    except Exception as e:
                        print(e)
                        print("loss : ", image_path)
            return True
        else:
            return False

    def extract_he_images(self):
        if not os.path.exists(os.path.join(self.workdir,"preprocessed")):
            os.mkdir(os.path.join(self.workdir,"preprocessed"))
        he_images_path = os.path.join(self.workdir,"preprocessed",self.imagedata,"he_images")
        ajax = os.path.join(self.workdir,"preprocessed",self.imagedata)
        if not os.path.exists(ajax):
            os.mkdir(ajax)
        if not os.path.exists(he_images_path):
            os.mkdir(he_images_path)

            labels = os.listdir(os.path.join(self.workdir,self.imagedata))
            for label in labels:
                os.mkdir(os.path.join(he_images_path,label))
                images = os.listdir(os.path.join(self.workdir,self.imagedata,label))
                for image in images:
                    try:
                        img_path = os.path.join(self.workdir,self.imagedata,label,image)
                        img = cv2.imread(img_path)
                        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                        he_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                        cv2.imwrite(os.path.join(he_images_path,label,"he_" + image),he_img)
                    except Exception as ex:
                        print("loss : ",image)
            return True
        else:
            return False
            #raise RuntimeError("Histogram eşitleme işlemi daha önce yapılmış!")

    def extract_clahe_images(self):
        if not os.path.exists(os.path.join(self.workdir,"preprocessed")):
            os.mkdir(os.path.join(self.workdir,"preprocessed"))
        he_images_path = os.path.join(self.workdir, "preprocessed", self.imagedata, "clahe_images")
        ajax = os.path.join(self.workdir,"preprocessed",self.imagedata)
        if not os.path.exists(ajax):
            os.mkdir(ajax)
        if not os.path.exists(he_images_path):
            os.mkdir(he_images_path)
            labels = os.listdir(os.path.join(self.workdir, self.imagedata))
            for label in labels:
                os.mkdir(os.path.join(he_images_path, label))
                images = os.listdir(os.path.join(self.workdir, self.imagedata, label))
                for image in images:
                    try:
                        img_path = os.path.join(self.workdir, self.imagedata, label, image)
                        img = cv2.imread(img_path)
                        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                        he_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                        cv2.imwrite(os.path.join(he_images_path, label,"clahe_" + image), he_img)
                    except Exception as ex:
                        print("loss : ", image)
            return True
        else:
            return False
            #raise RuntimeError("CLA Histogram eşitleme işlemi daha önce yapılmış!")