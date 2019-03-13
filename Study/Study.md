# 목차
[1. 머신러닝의 유형](#머신러닝의-유형)  

[2. 설치](#설치)  
[2-1. 파이썬](#파이썬)  
[2-2. 아나콘다](#아나콘다)  

[3. 파이썬 기본 문법](#파이썬-기본-문법)  
[3-1. 연산자](#연산자)  
[3-2. 자료형](#자료형)  
[3-3. 리스트](#리스트)  
[3-4. 딕셔너리](#딕셔너리)  
[3-5. bool](#bool)  
[3-6. 함수](#함수)  
[3-7. for](#for)  


# 머신러닝의 유형
## 1. 지도 학습 (Supervised Learning)
 : 훈련 데이터로 모델을 학습하여 경험하지 못한 데이터나 미래의 데이터에 관한 예측을 만드는 것.
## 2. 비지도 학습 (Unsupervised Learning)
 : 레이블이 없거나 구조를 모르는 데이터를 다뤄 결과변수나 보상함수의 도움 없이 데이터 구조를 탐색해 의미 있는 정보를 추출하는 것.
## 3. 강화 학습 (Reinforcement Learning)
 : 환경과의 상호작용을 기반으로 자신의 성능을 개선하는 시스템 개발.
 <br><br><br>

# 설치
## 파이썬
파이썬은 3.4.3 이상을 추천. 나는 3.7.2 최신 버전(2019년 3월 13일 기준)을 받았다.

그 다음 cmd에서 
pip install SomePackage
pip install SomePackage --upgrade
를 입력해 추가 파이썬 패키지를 설치, 업그레이드한다.

## 아나콘다

그리고 Anaconda를 받는다. 
데이터과학, 수학, 공학을 위한 파이썬의 모든 필수 요소들을 갖추고 있다.
https://www.anaconda.com/distribution/#download-section

그리고 아나콘다가 Code랑 파트너십을 맺고 있다길래 Visual Studio Code도 깔았다.

cmd에서 
conda install SomePackage
conda update SomePackage
를 해준다.


cmd에서
python --version
을 치면 python 3.7.1이라고 뜬다.


# 파이썬 기본 문법

## 연산자
+-*/ 는 그대로  
**는 거듭제곱 (3 ** 2 = 9)


## 자료형
int, float 등. str은 문자열


type(1) => int라고 출력  
type("xx") => str라고 출력  


print() <= 출력함수

x = 1  
print(x) <= 1이라고 출력



## 리스트
a = [1, 2, 3]  
print(a) <= [1, 2, 3] 출력  
len(a) <= 3 출력  
a[0] <= 1 출력  
a[0:2] <= [1, 2] 출력  
a[1:3] <= [2, 3] 출력  
a[:2] <= [1, 2] 출력  
a[2:] <= [3] 출력  
a[:-2] <= [1] 출력  
a[:-1] <= [1, 2] 출력



## 딕셔너리
a = {'q':1}  
a <= {'q':1} 출력  
a['q'] <= 1 출력  
a['w'] = 2  
print(a) <= {'q': 1, 'w': 2} 출력  



## bool
i = True  
you = False  
type(i) <= bool 출력  
i <= true 출력  
you = false 출력  
not i <= false 출력  
i and you <= false 출력  
i or you <= true 출력  



## 함수
def what(o):  
	print("the "+o)  
what("hell") <= the hell 출력



## for
for i in[3,2,1]:  
	print(i)  
<=3 2 1이 세 줄로 나뉘어 출력



