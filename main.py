#My Last File Died Man :/

import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import time

#Import Images
from keras.preprocessing.image import load_img
from keras.utils import to_categorical

def get_images(loc, name, start, amount, maximum = 7000, suffix = 'png'):
    
    output = []
    tempimg = None
    
    for i in range(start, start + amount):
        #Each Image
        tempimg = load_img(str(loc) + '/' + str(name) + ' (' + str(i % maximum + 1) + ').' + str(suffix))
        output.append(np.array(tempimg) / 255)
    
    return np.array(output)
    

def img2lab(image_set):
    image_set = np.clip((image_set * 255), 0, 18)[...,1]
    
    return to_categorical(image_set, num_classes = 19)

def get_labels(loc, name, start, amount, maximum = 7000, suffix = 'png'):
    return img2lab(get_images(loc, name, start, amount, maximum, suffix))

def count_classes(labels):
    values = np.zeros([19])
    
    for label in labels:
        print("Label!")
        for row in label:
            for col in row:
                for value in range(len(col)):
                    if col[value] > 0.5:
                        values[value] = values[value] + 0.1
    
    return values

def interpolate(arr, am = 10):
    
    am = max(am, 1)
    
    output = np.zeros([floor(len(arr)/am)])
    
    for i in range(len(output)):
        
        su = 0
        
        for j in range(i*am, i*am+am):
            
            su = su + arr[j]
            
        output[i] = su/am

    return output

train_prop = np.array([0.08254, 0.17132, 0.10844, 0.20595, 
                       0.17389, 0.2081, 0.33375, 0.16435, 
                       0.08348, 0.18097, 0.06372, 0.39058, 
                       0.64009, 0.12046, 0.24074, 0.167, 
                       0.74908, 0.99366, 0.01])
                


#Model
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Conv2D, LeakyReLU, concatenate, Conv2DTranspose, BatchNormalization
from keras.layers import Activation, MaxPooling2D, UpSampling2D, Dense, SpatialDropout2D
from keras.optimizers import Adam
from keras import backend as K

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)

class GAN(object):
    
    def __init__(self):
        
        #Autoencoder
        self.G = None
        
        #Old G
        self.OG = None
        
        #Labels To Color
        self.LTC = None
        
        #Learning Rate
        self.LR = 0.0003
        
        #Iterations
        self.steps = 0
        
    def generator(self):
        
        if self.G:
            return self.G
        
        #Input
        gi = Input(shape = [720, 1280, 3])
        
        #1280x720
        gc = Conv2D(filters = 32, kernel_size = 3, padding = 'same')(gi)
        gp = MaxPooling2D()(gc)
        ga = LeakyReLU()(gp)
        #640x360
        gt1 = Activation('linear')(ga)
        gc = Conv2D(filters = 32, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #640x360
        gc = Conv2D(filters = 32, kernel_size = 3, padding = 'same')(ga)
        gp = MaxPooling2D()(gc)
        ga = LeakyReLU()(gp)
        #320x180
        gt2 = Activation('linear')(ga)
        gc = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #320x180
        gc = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(ga)
        gp = MaxPooling2D()(gc)
        ga = LeakyReLU()(gp)
        #160x90
        gt3 = Activation('linear')(ga)
        gc = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #160x90
        gc = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(ga)
        gp = MaxPooling2D()(gc)
        ga = LeakyReLU()(gp)
        #80x45
        gt4 = Activation('linear')(ga)
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #80x45
        gc = Conv2D(filters = 192, kernel_size = 5, padding = 'same')(ga)
        gp = MaxPooling2D((5, 5))(gc)
        ga = LeakyReLU()(gp)
        #16x9
        
        #ENCODED
        
        
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        
        
        #DECODE
        
        #16x9
        gc = Conv2DTranspose(filters = 192, kernel_size = 5, padding = 'same')(ga)
        gu = UpSampling2D((5,5))(gc)
        ga = LeakyReLU()(gu)
        #80x45
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gm = concatenate([ga, gt4])
        #80x45
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(gm)
        ga = LeakyReLU()(gc)
        #80x45
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #160x90
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gm = concatenate([ga, gt3])
        #160x90
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(gm)
        ga = LeakyReLU()(gc)
        #160x90
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #320x180
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gm = concatenate([ga, gt2])
        #320x180
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(gm)
        ga = LeakyReLU()(gc)
        #320x180
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #640x360
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gm = concatenate([ga, gt1])
        #640x360
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(gm)
        ga = LeakyReLU()(gc)
        gb = BatchNormalization(momentum = 0.9)(ga)
        #640x360
        gc = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(gb)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #1280x720
        gc = Conv2DTranspose(filters = 64, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #OUTPUT
        go = Dense(19, activation = 'softmax')(ga)
        
        self.G = Model(inputs = gi, outputs = go)
        
        return self.G
    
    def GenModel(self):
        
        if self.G:
            self.G.compile(optimizer = Adam(lr = self.LR / pow(5/4, round(self.steps / 5000))),
                           loss = 'categorical_crossentropy')
            return self.G
        
        self.generator()
        self.G.compile(optimizer = Adam(lr = self.LR / pow(5/4, round(self.steps / 5000))),
                        loss = 'categorical_crossentropy')
        
        return self.G
    
    def Lab_To_Col(self):
        
        if self.LTC:
            return self.LTC
        
        #Generator
        ltc_file = open("ltc/ltc.json", 'r')
        ltc_json = ltc_file.read()
        ltc_file.close()
        
        self.LTC = model_from_json(ltc_json)
        self.LTC.load_weights("ltc/ltc.h5")
        
        return self.LTC
    
    def save_OG(self):
        
        self.OG = self.G.get_weights()
        
        return self.OG
    
    def load_OG(self):
        
        self.G.set_weights(self.OG)
        
        return self.G
        
        







class GModel(object):
    
    def __init__(self, steps = 0, loss = []):
        
        self.GAN = GAN()
        self.GenModel = self.GAN.GenModel()
        self.LTC = self.GAN.Lab_To_Col()
        
        self.GAN.steps = steps
        
        self.loss = loss
        
    def train(self, batch = 2):
        
        #Train Encoder
        loss = self.train_gen(batch)
        
        self.loss.append(loss)
        
        #Clear Memory
        #if self.GAN.steps % 10 == 0:
            #self.reset_memory()
        
        if self.GAN.steps % 50 == 0:
            self.save()
            self.evaluate()
            
        self.GAN.steps = self.GAN.steps + 1
        
    def train_gen(self, batch = 2):
        
        
        
        star = random.randint(0, 7000)
        
        x = get_images('data/images', 'img', star, batch, suffix = 'jpg')
        y = get_labels('data/labels', 'seg', star, batch, suffix = 'png')
        
        loss = self.GenModel.train_on_batch(x, y, class_weight = train_prop)
        
        #Evaluate
#        star = random.randint(0, 999)
#        
#        x = get_images('val/images', 'img', star, batch, maximum = 999, suffix = 'jpg')
#        y = get_labels('val/labels', 'seg', star, batch, maximum = 999, suffix = 'png')
#        
#        loss = self.GenModel.test_on_batch(x, y)
        
        print(loss)
        
        
        
        return loss
    
    def save(self, num = -1):
        
        if num == -1:
            num = floor(self.GAN.steps/300)
        
        gen_json = self.GAN.G.to_json()

        with open("Models/gen.json", "w") as json_file:
            json_file.write(gen_json)

        self.GAN.G.save_weights("Models/gen"+str(num)+".h5")
        
        return True
    
    def load(self, num = 0):
        
        steps1 = self.GAN.steps
        
        self.GAN = GAN()
        
        gen_file = open("Models/gen.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()
        
        self.GAN.G = model_from_json(gen_json)
        self.GAN.G.load_weights("Models/gen"+str(num)+".h5")
        
        #Reinitialize
        self.GAN.steps = steps1
        self.GenModel = self.GAN.GenModel()
        
        return True
    
    def reset_memory(self):
        
        self.save('_tmp')
        steps = self.GAN.steps
        
        K.clear_session()
        
        self.__init__(steps, loss = self.loss)
        self.load('_tmp')
        
    def evaluate(self):
        
        #Choose Image
        star = random.randint(0, 999)
        
        x = get_images('val/images', 'img', star, 1, suffix = 'jpg')
        
        y1 = get_images('val/color', 'seg', star, 1, suffix = 'png')
        
        y2 = self.GenModel.predict(x)
        y2 = self.LTC.predict(y2)
        
        plt.figure(1)
        plt.plot(interpolate(self.loss, round(len(self.loss) / 50)))
        #plt.savefig('LossGraph/loss_' + str(floor(self.GAN.steps / 500)) + '.png')
        plt.figure(2)
        plt.imshow(x[0])
        plt.figure(3)
        plt.imshow(y1[0])
        plt.figure(4)
        plt.imshow(y2[0])
        
        
        plt.show(block=False)
        plt.pause(0.01)
        

model = GModel(100000)
model.load(291)


while(True):
    print("\n Round " + str(model.GAN.steps) + ":")
    
    model.evaluate(1)
    
    time.sleep(0.05)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        