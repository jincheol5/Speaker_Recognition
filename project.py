import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


#np.transpose
#########################################################
#음성 파일 mfcc 추출하기

#Train File Load(16000Hz/16bits/Mono)
audio1,sr1=librosa.load('data/F1.wav',sr=16000)


#각 화자 음성파일 MFCC 추출,2차원 배열로 저장된다  n_fft=frame 몇으로 나눌지 
mfcc1=librosa.feature.mfcc(y=audio1,sr=sr1,n_mfcc=24,n_fft=500)

audio1,sr1=librosa.load('test_data/1.wav',sr=16000)


t_mfcc1=librosa.feature.mfcc(y=audio1,sr=sr1,n_mfcc=24,n_fft=500)

#편리한 data handling을 위해 pandas data frame으로 변환 
mfccDF=pd.DataFrame(data=mfcc1)

#print(mfccDF)

#n_mfcc=음성파일 몇으로 나눌지,n_fft=몇개로 프레임 나눌지
#########################################################

gmm = GaussianMixture(n_components=5)

gmm.fit(mfccDF)


#gmm.predict(mfccDF)

#gmm.score(t_mfcc1) #score 안에는 testdata의 mfcc 값을 넣는다  결과값은 log를 한  확률값이 나온다 


print(gmm.score(t_mfcc1))




#n_components = mixture 개수 
#gmm_cluster_labels = gmm.predict(mfcc1) #트레이닝
#입력 데이터 = mfcc
