#!/usr/bin/python
# -*- coding:UTF-8 -*-

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import warnings
import math
import os

class RandomForest:
    # 初始化，神经网络参数，树的数量，数据集路径，训练集与测试集的比例
    def __init__(self,input,dataPath='./data/offline/',pct=0.9):
        self.input = input
        self.pct = pct
        self.data = self.get_data(dataPath)
        self.init_data()
        self.rf = RandomForestRegressor(n_estimators=1000,random_state=0)  # 一般来说n_estimators越大越好


    def get_data(self,dataPath):
        # 训练的文件全部在data文件夹下，datapath为data的路径，读取该文件夹下的数据文件
        filePath = []
        pathDir = os.listdir(dataPath)
        for allDir in pathDir:
            child = os.path.join('%s%s' % (dataPath, allDir))
            filePath.append(child)
        data = []
        for path in filePath:
            fopen = open(path, 'r')
            for eachLine in fopen:
                eachLineData = eachLine.split(",")
                eachLineData = np.array(list(map(lambda x:float(x),eachLineData)))
                data.append(eachLineData)
            fopen.close()
        return np.array(data)

    # 按照比例分隔数据集，构造训练集和测试集
    def init_data(self):
        # self.data = self.data[0:100,:]
        self.trainX = self.data[:int(len(self.data)*self.pct),:self.input]
        self.trainY = self.data[:int(len(self.data)*self.pct),self.input:self.input+2]
        self.testX = self.data[int(len(self.data)*self.pct):,:self.input]
        self.testY = self.data[int(len(self.data)*self.pct):,self.input:self.input+2]
        print(self.trainX.shape)
        print(self.trainY.shape)

    # 离线训练: EPOCH代表迭代次数（暂时不知道迭代次数是否起作用，默认为1）
    def offline_train(self):
        print("Train start...")
        self.rf.fit(self.trainX,self.trainY)
        rmse,score = self.test()
        print("Train finish!")
        print("RMSE: " + str(np.average(rmse)))
        print("Score:" + str(score))
        self.save()
        print("Save model!")
        print()

    # 在线训练: newData代表新产生的data,EPOCH代表迭代次数（暂时不知道迭代次数是否起作用，默认为1），flag为true代表保留旧数据，false代表替换旧数据
    def online_train(self,newDataPath="./data/online/",flag=True):
        # 加载模型
        self.load()
        print("Online train start...")
        newData = self.get_data(newDataPath)
        if flag:
            self.data = np.concatenate([self.data,newData],0)
        else:
            self.data = newData
        self.init_data()
        self.offline_train()
        print("Online train finish!")
        self.save()
        print("Save model!")
        print()

    def save(self):
        # 保存Model(注:save文件夹要预先建立，否则会报错)
        joblib.dump(self.rf, "./save/rf.pkl")

    def load(self):
        # 读取Model
        self.rf = joblib.load("./save/rf.pkl")

    def test(self):
        preds = self.rf.predict(self.testX)
        # RMSE 均方根误差亦称标准误差
        sse = sum(map(lambda z: (z[0] - z[1]) * (z[0] - z[1]), zip(preds, self.testY)))
        rmse = np.sqrt(sse / float(len(preds)))
        score = self.rf.score(self.testX, self.testY)
        return rmse,score

    def predict(self,testX):
        preds = self.rf.predict(testX)
        return preds


if __name__ == "__main__":
    # 忽略一些版本不兼容等警告
    warnings.filterwarnings("ignore")
    orf = RandomForest(input=722)
    # orf.offline_train()
    orf.online_train()
# RMSE: 0.036099242494643385
# Score:-2.9361148161707598
