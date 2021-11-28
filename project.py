import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

#n_mfcc=음성파일 몇으로 나눌지,n_fft=몇개로 프레임 나눌지

def make_gmm():
  #각 음성파일의 gmm들의 리스트 반환 
  
  path="data/"
  file_name=['F1','F2','M1','M2']
  
  gmm_list=[]
  
  #Train File Load(16000Hz/16bits/Mono)
  for i in range(4):
    audio,sr=librosa.load(path+file_name[i]+'.wav',sr=16000)
    
    #각 화자 음성파일 MFCC 추출,2차원 배열로 저장된다  n_fft=frame 몇으로 나눌지 
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=24,n_fft=500)
    mfcc=np.transpose(mfcc)
    
    gmm = GaussianMixture(n_components=5) #mixture =5
    gmm.fit(mfcc) #학습
    
    gmm_list.append(gmm)

  return gmm_list

def test_gmm(gmm_list,number):
  
  path="test_data/"
  
  speaker=""
  
  test_score=[]
  
  t_audio,t_sr=librosa.load(path+number+'.wav',sr=16000)
  
  t_mfcc=librosa.feature.mfcc(y=t_audio,sr=t_sr,n_mfcc=24,n_fft=500)
  t_mfcc=np.transpose(t_mfcc)
  
  for i in range(4):
    test_score.append(gmm_list[i].score(t_mfcc))
  
  if max(test_score)==test_score[0]:
    print('화자 = F1')
    speaker='F1'
  elif max(test_score)==test_score[1]:
    print('화자 = F2')
    speaker='F2'
  elif max(test_score)==test_score[2]:
    print('화자 = M1')
    speaker='M1'
  else:
    print('화자 = M2')
    speaker='M2'
    
  return speaker
  
  
  
gmm_list=make_gmm()


while(True):
  print("test 파일의 번호를 입력해주세요(1,2,3,4,5) / 종료는 0 입력")
  test_num=input("test_number : ")
  if test_num=='0':
    print("화자 인식 종료") 
    break
  speaker=test_gmm(gmm_list,test_num) #speaker = 인식된 화자 

















#gmm.predict(mfccDF)

#gmm.score(t_mfcc1) #score 안에는 testdata의 mfcc 값을 넣는다  결과값은 log를 한  확률값이 나온다 







#n_components = mixture 개수 
#gmm_cluster_labels = gmm.predict(mfcc1) #트레이닝
#입력 데이터 = mfcc
