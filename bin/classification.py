import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np
import cv2
import shutil
import seaborn as sns

from keras.utils import to_categorical
import keras
from keras.applications import vgg16,resnet50,inception_v3
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle

from bin.data_informations_enum import DataInformations

class Classification:
    def __init__(self,model_dict):
        self.model_dict = model_dict

        self.global_image_data = []
        self.global_label_data = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.global_epoch_lim = model_dict["epochLim"]

        self.global_image_size = (150,150)
        self.global_labels = os.listdir(os.path.join(self.model_dict["workdir"],"images"))
        self.num_classes = len(self.global_labels)
        self.model = None


    def start(self):
        original_data, original_labels = self.get_image_data()
        original_he_data,original_he_labels = self.get_image_data(data_path="preprocessed/images/he_images")
        original_clahe_data,original_clahe_labels = self.get_image_data(data_path="preprocessed/images/clahe_images")

        self.global_image_data = np.concatenate([original_data,original_he_data,original_clahe_data])
        self.global_label_data = np.concatenate([original_labels,original_he_labels,original_clahe_labels])

        if self.model_dict["data_informations"]["Dataset"] == DataInformations.DATASET_AUGMENTATED_AND_ORIGINAL.value:
            augmentated_data, augmentated_labels = self.get_image_data()
            augmentated_he_data, augmentated_he_labels = self.get_image_data(data_path="preprocessed/augmentated_images/he_images")
            augmentated_clahe_data, augmentated_clahe_labels = self.get_image_data(data_path="preprocessed/augmentated_images/clahe_images")
            self.global_image_data = np.concatenate([self.global_image_data,augmentated_data,augmentated_he_data,augmentated_clahe_data])
            self.global_label_data = np.concatenate([self.global_label_data,augmentated_labels,augmentated_he_labels,augmentated_clahe_labels])

        self.shuffle_data()
        self.image_data_split()
        if self.model_dict["data_informations"]["ClassifierAlgorithm"] == DataInformations.CLASSIFIER_RF.value:
            self.create_rf_model()
            self.rf_model_fit()
            self.classification_rep = self.get_rf_confusion_matrix()
            plot_path = os.path.join(self.model_dict["workdir"], "plots")
            if os.path.exists(plot_path):
                shutil.rmtree(plot_path)
            os.mkdir(plot_path)
            plt.figure(figsize=(5, 3))
            sns.set(font_scale=1.2)
            ax = sns.heatmap(self.classification_rep["confusion"], annot=True, xticklabels=self.global_labels,
                             yticklabels=self.global_labels, cbar=False,
                             cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
            plt.yticks(rotation=0)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            ax.xaxis.set_ticks_position('top')
            plt.title('Confusion matrix')
            plt.savefig(os.path.join(plot_path, "rf_confusion_matrix.png"))
            plt.cla()
            plt.clf()
            plt.figure()
            plt.plot(self.classification_rep["fpr"]["micro"], self.classification_rep["tpr"]["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(self.classification_rep["roc_auc"]["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(self.classification_rep["fpr"]["macro"], self.classification_rep["fpr"]["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(self.classification_rep["roc_auc"]["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(len(self.y_test[0])), colors):
                plt.plot(self.classification_rep["fpr"][i], self.classification_rep["tpr"][i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, self.classification_rep["roc_auc"][i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plot_path, "rf_roc_graph.png"))
            plt.cla()
            plt.clf()

        elif self.model_dict["data_informations"]["ClassifierAlgorithm"] == DataInformations.CLASSIFIER_ANN.value:
            self.create_model()
            self.model_fit()
            self.classification_rep = self.get_confusion_matrix()

            plot_path = os.path.join(self.model_dict["workdir"], "plots")
            if os.path.exists(plot_path):
                shutil.rmtree(plot_path)

            os.mkdir(plot_path)

            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            plt.savefig(os.path.join(plot_path,"accuracy_graph.png"))
            plt.cla()
            plt.clf()

            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(plot_path, "loss_graph.png"))
            plt.cla()
            plt.clf()

            plt.figure(figsize=(5, 3))
            sns.set(font_scale=1.2)
            ax = sns.heatmap(self.classification_rep["confusion"], annot=True, xticklabels=self.global_labels, yticklabels=self.global_labels, cbar=False,
                             cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
            plt.yticks(rotation=0)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            ax.xaxis.set_ticks_position('top')
            plt.title('Confusion matrix')
            plt.savefig(os.path.join(plot_path,"confusion_matrix.png"))
            plt.cla()
            plt.clf()

            plt.figure()
            plt.plot(self.classification_rep["fpr"]["micro"], self.classification_rep["tpr"]["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(self.classification_rep["roc_auc"]["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(self.classification_rep["fpr"]["macro"], self.classification_rep["fpr"]["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(self.classification_rep["roc_auc"]["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(len(self.y_test[0])), colors):
                plt.plot(self.classification_rep["fpr"][i], self.classification_rep["tpr"][i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, self.classification_rep["roc_auc"][i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plot_path,"roc_graph.png"))
            plt.cla()
            plt.clf()
        self.global_image_data = []
        self.global_label_data = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def get_image_data(self, data_path = "images"):
        data = []
        labels = []
        original_path = os.path.join(self.model_dict["workdir"],data_path)
        original_labels = os.listdir(original_path)
        for label_no,original_label in enumerate(original_labels):
            images_path = os.path.join(original_path,original_label)
            image_names = os.listdir(images_path)
            for image_name in image_names:
                try:
                    image_path = os.path.join(images_path,image_name)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image_from_array = Image.fromarray(image,'RGB')
                    resized_image = image_from_array.resize(self.global_image_size)
                    data.append(np.array(resized_image))
                    labels.append(np.array(label_no))
                except Exception as ex:
                    print(ex)
                    print("Loss : ", image_name)
        return np.array(data),np.array(labels)

    def shuffle_data(self):
        s = np.arange(self.global_image_data.shape[0])
        np.random.shuffle(s)
        self.global_image_data = self.global_image_data[s]
        self.global_label_data = self.global_label_data[s]

    def image_data_split(self):
        data_length = len(self.global_image_data)
        self.X_train,self.X_test = self.global_image_data[(int)(0.2*data_length):],self.global_image_data[:(int)(0.2*data_length)]
        self.X_train = self.X_train.astype("float32")/255
        self.X_test = self.X_test.astype("float32")/255

        self.y_train,self.y_test = self.global_label_data[(int)(0.2*data_length):],self.global_label_data[:(int)(0.2*data_length)]
        self.y_train = to_categorical(self.y_train,self.num_classes)
        self.y_test = to_categorical(self.y_test,self.num_classes)

    def create_model(self):
        base_model = None
        if self.model_dict["data_informations"]["DeepLearningAlgorithm"] == DataInformations.DL_VGG16.value:
            print("vgg16")
            base_model = vgg16.VGG16(weights = "imagenet", include_top = False, input_shape = (self.global_image_size[0],self.global_image_size[1],3))
        elif self.model_dict["data_informations"]["DeepLearningAlgorithm"] == DataInformations.DL_RESNET50.value:
            print("resnet50")
            base_model = resnet50.ResNet50(weights = "imagenet", include_top = False, input_shape = (self.global_image_size[0],self.global_image_size[1],3))
        base_model.trainable = False
        print("-----------------------")
        print(base_model.summary())
        print("-----------------------")
        inputs = keras.Input(shape=(self.global_image_size[0], self.global_image_size[1], 3))
        x = base_model(inputs, training=False)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(500, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        self.model = model
        print(model.summary())
        print("----------------------")

    def model_fit(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)
        history = self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=self.global_epoch_lim, verbose=1, callbacks=[callback],
                            validation_data=(self.X_test, self.y_test))
        self.history = history


    def create_rf_model(self):
        base_model = None
        if self.model_dict["data_informations"]["DeepLearningAlgorithm"] == DataInformations.DL_VGG16.value:
            base_model = vgg16.VGG16(weights="imagenet", include_top=False,
                                     input_shape=(self.global_image_size[0], self.global_image_size[1], 3))
        elif self.model_dict["data_informations"]["DeepLearningAlgorithm"] == DataInformations.DL_RESNET50.value:
            base_model = resnet50.ResNet50(weights="imagenet", include_top=False,
                                           input_shape=(self.global_image_size[0], self.global_image_size[1], 3))
        for layer in base_model.layers:
            layer.trainable = False
        X_train_feature_extractor = base_model.predict(self.X_train)
        self.X_train_features = X_train_feature_extractor.reshape(X_train_feature_extractor.shape[0],-1)

        X_test_feature_extractor = base_model.predict(self.X_test)
        self.X_test_features = X_test_feature_extractor.reshape(X_test_feature_extractor.shape[0], -1)

        self.TL_extractor = base_model
        self.model = RandomForestClassifier(n_estimators=50,random_state=42)

    def rf_model_fit(self):
        self.model.fit(self.X_train_features,self.y_train)

    def get_confusion_matrix(self): #Global function
        y_pred = self.model.predict(self.X_test,batch_size = 32)
        y_pred = np.argmax(y_pred,axis=1)
        y_test_cm = np.argmax(self.y_test,axis=1)
        conf = confusion_matrix(y_test_cm,y_pred)
        clrp = classification_report(y_test_cm,y_pred,target_names=self.global_labels)

        #roc curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_pred_e = to_categorical(y_pred)
        print(self.y_test[0])
        print(len(self.y_test[0]))
        for i in range(len(self.y_test[0])):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_e[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), y_pred_e.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.y_test[0]))]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(self.y_test[0])):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= len(self.y_test[0])
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return {
            "confusion" : conf,
            "classification_report" : clrp,
            "fpr" : fpr,
            "tpr" : tpr,
            "roc_auc" : roc_auc
        }

    def classification_report(self):
        #Confusion matrix
        conf_matrix_str = "\n\nKarışıklık Matrisi\n------------------\n"
        cm = self.classification_rep["confusion"]
        for i in range(cm.shape[0]):
            conf_matrix_str += "[ "
            for j in range(cm.shape[1]):
                if j != cm.shape[1] - 1:
                    conf_matrix_str += str(cm[i, j]) + ", "
                else:
                    conf_matrix_str += str(cm[i, j]) + " "
            conf_matrix_str += "]\n"

        #Classification report
        cr = "\n\nSINIFLANDIRMA RAPORU\n---------------------\n"
        cr += self.classification_rep["classification_report"]

        #Epoch steps
        epoch_steps_str = ""
        if self.model_dict["data_informations"]["ClassifierAlgorithm"] == DataInformations.CLASSIFIER_ANN.value:
            epoch_steps_str = "\n\nEpoch başına Accuracy - Loss oranları\n--------------------------------------\n"
            count = len(self.history.history["accuracy"])
            for i in range(count):
                epoch_steps_str += "Adım " + str(i + 1) + "/" + str(count) + " -> Accuracy : " + str(
                    self.history.history["accuracy"][i]) + ", Loss : " + str(self.history.history["loss"][i]) + "\n"
        return {
            "confusion_matrix_str" : conf_matrix_str,
            "classification_report_str" : cr,
            "epoch_steps_str" : epoch_steps_str
        }

    def get_rf_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test_features)
        y_train_decoded = np.argmax(self.y_train, axis=1)
        y_test_decoded = np.argmax(self.y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        conf = confusion_matrix(y_test_decoded,y_pred)
        clrp = classification_report(y_test_decoded,y_pred)
        # roc curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_pred_e = to_categorical(y_pred)

        for i in range(len(self.y_test[0])):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_e[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), y_pred_e.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.y_test[0]))]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(self.y_test[0])):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= len(self.y_test[0])
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return {
            "confusion": conf,
            "classification_report": clrp,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc
        }

    def predict(self,im):
        img = cv2.imread(im)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img,'RGB')
        image = img.resize(self.global_image_size)
        image = np.array(image)
        image = image / 255
        if self.model_dict["data_informations"]["ClassifierAlgorithm"] == DataInformations.CLASSIFIER_ANN.value:
            a = []
            a.append(image)
            a = np.array(a)
            score = self.model.predict(a,verbose=1)
            label_index = np.argmax(score)
            acc = np.max(score)
            weather = self.global_labels[int(label_index)]
            return weather,round(acc*100)
        else:
            input_img = np.expand_dims(image,axis=0)
            input_img_feature = self.TL_extractor.predict(input_img)
            input_img_feature = input_img_feature.reshape(input_img_feature.shape[0],-1)
            prediction_RF_ARR = self.model.predict(input_img_feature)
            prediction_RF = prediction_RF_ARR[0]
            label_index = np.argmax(prediction_RF)
            weather = self.global_labels[int(label_index)]
            return weather,-1

