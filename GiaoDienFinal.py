#Import thư viện giao diện
import sys
import pathlib
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
import tkinter
from tkinter import ttk
from tkinter.filedialog import Open, SaveAs
from tkinter import messagebox

#Import thư viện train
import cv2
from matplotlib.ft2font import BOLD
from matplotlib.pyplot import bar_label, text
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.models import load_model
import tensorflow as tf
import math as m
from yaml import load

#Thư viện cho convert spectrogram
import cv2
import glob
import numpy as np
import pandas as pd
import os
import wave
import pylab
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt

#Load model
classifier = load_model('C:/Users/levun/Desktop/Project_Final_Ai/CNN_final.h5')

#Tạo lớp class
class Example(Frame):
    def __init__(self, parent):
        tkinter.Frame.__init__(self, parent, background="white")
    
        self.parent = parent
        self.initUI()
    
    def initUI(self):
        self.parent.title("Nguyen Le Vu_19146428")
        self.style = Style()
        self.style.theme_use("default")
        self.pack(fill=BOTH, expand=1)

        #Tạo Label
        lbl = tkinter.Label(self, text = "PROJECT FINAL AI", fg="dark blue", bg="white", font=("Arial",20,"bold"))
        lbl.pack(side = TOP, padx=15,pady=15)

        lbl1 = tkinter.Label(self, text = "RECOGNIZE 10 ANIMAL BY SOUND", fg="dark blue", bg="white", font=("Arial",20,"bold"))
        lbl1.pack(side = TOP,padx=0,pady=0)

        lbl2 = tkinter.Label(self, text="Please Choose Mode:",fg='#442265', bg="white", font=("Verdana",14))
        lbl2.place(x=0,y=150, width=450,height=25)

        #Tạo Nút Nhấn
        ConvertButton = tkinter.Button(self, text="Convert", bg="#FFFF99",activebackground="white",font=("Verdana",16), command=self.onConvert)
        ConvertButton.place(x=45, y=210, width=100,height=40)

        loadButton = tkinter.Button(self, text="Load", bg="Light Blue",activebackground="white",font=("Verdana",16), command=self.onLoad)
        loadButton.place(x=150, y=210, width=80,height=40)

        RegButton = tkinter.Button(self, text="Recognize", bg="Light green", activebackground="white",font=("Verdana",16), command=self.onTest)
        RegButton.place(x=235, y=210, width=120,height=40)

        exitButton = tkinter.Button(self, text="Exit",bg="pink", activebackground="white",font=("Verdana",16), command=self.onExit)
        exitButton.place(x=360, y=210, width=80,height=40)

    #Load ảnh 
    def onLoad(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png'),('All', '*.all')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global imgin
            #imgin = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)
            imgin = cv2.resize(imgin, (250,250))
            cv2.namedWindow("Image_Test", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Image_Test", imgin)
            lbl2 = tkinter.Label(self, text="Recognize Object Are:",fg='#442265', bg="white", font=("Verdana",14))
            lbl2.place(x=0,y=150, width=350,height=25)
            lbl3 = tkinter.Label(self, text="",fg='#442265', bg="white", font=("Verdana",14))
            lbl3.place(x=280,y=150, width=100,height=25)

    #Nút thoát
    def onExit(self):
        msg = messagebox.showinfo( "Note","Do you exit?")
        self.quit()

    # Nhận diện
    def onTest(self):
        img = cv2.resize(imgin, (250,250))
        img = np.array(img)
        img = img.reshape(1,250,250,3)
        img = img.astype('float32')
        img /=255
        pred = np.argmax(classifier.predict(img),axis = -1)
        label = ['Bird', 'Cat', 'Cow', 'Cricket', 'Dog', 'Gecko', 'Goat', 'Horse', 'Lion', 'Squirrel']
        label[pred[-1]]
        lbl3 = tkinter.Label(self, text=""+ label[np.argmax(classifier.predict(img.reshape(1,250,250,3)))],fg='#442265', bg="white", font=("Verdana",14))
        lbl3.place(x=280,y=150, width=100,height=25)


    # Convert âm thanh sang ảnh Spectrogram
    def onConvert(self):

        #Hàm load file mp3
        def get_wav_info(wav_file):
            wav = wave.open(wav_file, 'r')
            frames = wav.readframes(-1)
            sound_info = pylab.fromstring(frames, 'int16')
            frame_rate = wav.getframerate()
            wav.close()
            return sound_info, frame_rate

        # Hàm convert sang Spectrogram 
        def graph_spectrogram(wav_file):
            sound_info, frame_rate = get_wav_info(wav_file)
            fig = pylab.figure(num=None, figsize=(13,6))
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            pylab.subplot(111)
            #pylab.title('spectrogram of %r' % wav_file)
            pylab.specgram(sound_info, Fs=frame_rate)
        #Mở file mp3 và xử  lý lệnh
        global ftypes
        ftypes = [('Sound', '*.wav *.mp3'),('All', '*.all')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
        signal, sr = librosa.load(fl, duration=10) 
        #Hàm chuyển đổi
        get_wav_info(fl)
        graph_spectrogram(fl)
        #lưu ảnh
        plt.savefig('C:/Users/levun/Desktop/Project_Final_Ai/data/Data convert/test.png', bbox_inches='tight')
        plt.show()
        lbl2 = tkinter.Label(self, text="Please Choose Mode:",fg='#442265', bg="white", font=("Verdana",14))
        lbl2.place(x=0,y=150, width=450,height=25)       
        
#Tạo window
window = Tk()
window.title("Giao Dien Project")
#Gọi class chứa các hàm def
app = Example(window)
window.geometry("480x260+200+250")
window.mainloop()

