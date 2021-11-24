import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#########################################################
#음성 파일 mfcc 추출하기

#Train File Load(16000Hz/16bits/Mono)
audio1,sr1=librosa.load('data/F1.wav',sr=16000)
audio2,sr2=librosa.load('data/F2.wav',sr=16000)
audio3,sr3=librosa.load('data/M1.wav',sr=16000)
audio4,sr4=librosa.load('data/M2.wav',sr=16000)

#각 화자 음성파일 MFCC 추출,2차원 배열로 저장된다 
mfcc1=librosa.feature.mfcc(y=audio1,sr=sr1,n_mfcc=12,n_fft=500)
mfcc2=librosa.feature.mfcc(y=audio2,sr=sr2,n_mfcc=12,n_fft=500)
mfcc3=librosa.feature.mfcc(y=audio3,sr=sr3,n_mfcc=12,n_fft=500)
mfcc4=librosa.feature.mfcc(y=audio4,sr=sr4,n_mfcc=12,n_fft=500)
#n_mfcc=음성파일 몇으로 나눌지,n_fft=몇개로 프레임 나눌지
#########################################################

gmm1 = GaussianMixture(n_components=5).fit(mfcc1)
gmm1 = GaussianMixture(n_components=5).fit(mfcc2)
gmm1 = GaussianMixture(n_components=5).fit(mfcc3)
gmm1 = GaussianMixture(n_components=5).fit(mfcc4)



audio1,sr1=librosa.load('test_data/1.wav',sr=16000)
audio2,sr2=librosa.load('test_data/2.wav',sr=16000)
audio3,sr3=librosa.load('test_data/3.wav',sr=16000)
audio4,sr4=librosa.load('test_data/4.wav',sr=16000)

t_mfcc1=librosa.feature.mfcc(y=audio1,sr=sr1,n_mfcc=12,n_fft=500)
t_mfcc2=librosa.feature.mfcc(y=audio2,sr=sr2,n_mfcc=12,n_fft=500)
t_mfcc3=librosa.feature.mfcc(y=audio3,sr=sr3,n_mfcc=12,n_fft=500)
t_mfcc4=librosa.feature.mfcc(y=audio4,sr=sr4,n_mfcc=12,n_fft=500)


#n_components = mixture 개수 
#gmm_cluster_labels = gmm.predict(mfcc1) #트레이닝
#입력 데이터 = mfcc

print()


print(gmm1.score(t_mfcc1))
print(gmm1.score(t_mfcc2))
print(gmm1.score(t_mfcc3))
print(gmm1.score(t_mfcc4))