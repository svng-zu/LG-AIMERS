# LG AIMERS MQL 데이터 기반 B2B 영업기회 창출 예측 모델 개발
<br/>

## 1. 개요
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/f4681e5c-4aca-4dfa-b662-43d839d82d16" alt="preview" width="80%" height="80%">
<br/>

## 2. EDA
### 전체 데이터 확인
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/9bae032e-b21f-42cf-bb43-fda2c30f461e" alt="data1" width="30%" height="30%">

<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/06ec7602-f1b7-4223-9e87-240d15f620f9" alt="data1" width="30%" height="30%">
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/fc7496a8-71a4-4e15-9745-3ab54bb6fa59" alt="data1" width="50%" height="50%">

#### 타겟 컬럼인 is_converted 열의 True와 False의 비율이 약 11: 1로 불균형이 있음을 알 수 있음


### 범주형 변수
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/c3bd18ae-8371-4e30-8cad-2c185750fd4c" alt="data" width=30% height=30%>
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/ee3fd6df-0429-46b3-927c-ac8293a9e9bf" alt="data" width=30% height=30%>
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/fdb061c3-9a6c-4c4c-aa0f-f3c916246ef3" alt="data" width=30% height=30%>



### 수치형 변수
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/ccff87f9-e838-4de0-aec4-47fe1caab7e8" alt="data" width=50% height=50%>

### 상관계수
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/5d40481f-2a2b-4191-be82-bd957a022311" alt="data" width=50% height=50%>

## 3. 데이터 전처리

### 컬럼 삭제
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/9049ec65-a8b9-4362-b942-d3723b4bc1f9" alt="data" width=80% height=80%>

<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/0b20e0d1-c783-4a7f-a6d9-bcb66c09792e" alt="data" width=70% height=70%>

#### 같은 의미의 다른 데이터는 같은 범주로 처리
-> etc, other, others -> etc
-> end-customer, end customer, end-user 등

#### 개수가 1개인 범주들을 기타 처리
-> 결측치와는 다르게 처리

#### 결측치 처리
-> 수치형 데이터는 0 대체해도 무방 했음
-> 범주형은 None이라는 문자열로 범주처럼 처리

<br/>
## 3. 모델링
### 모델 선택
#### autoML - pycaret 사용

pycaret
ML workflow을 자동화 하는 opensource library로 여러 머신러닝 task에서 사용하는 모델들을 하나의 환경에서 비교하고 튜닝하는 등 간단한 코드를 통해 편리하게 사용할 수 있도록 자동화환 라이브러리


# autoML 실행결과
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/1571c80d-74b2-40ce-bf06-91e4caea475e" alt="data" width=70% height=70%>

각 모델에 대해서 어떤 모델을, 몇 개를 조합할 것인지에 대한 실험이 필요

<br/>
## 4. 과적합 핸들링
<br/>
### 1. 언더샘플링
앞선 타겟 컬럼인 is_converted의 True와 False 값의 비율이 약 11:1로 클래스 불균형이 심한 상태.
이를 그대로 학습하게 되면 False 클래스에 편향된 모델이 되기 때문에 오버 샘플링 / 언더 샘플링을 진행
실험 결과 언더 샘플링의 F1-score가 더 높아 언더 샘플링을 진행
정보 손실의 위험 ---> 앙상블 + 보팅으로 해결
public score 0.6

### 2. 앙상블
여러개의 예측 모델을 결합하여 과적합을 줄이고 모델을 일반화하는 방법
앞서 고른 상위 5개 모델을 앙상블하여 모델 일반화 진행 함.

### 3. 모델 학습 시 편향되어 학습되는 요인 찾기
train 데이터에서 customer_idx = 25096 의 경우 영업 횟수 2421 모두 성공한 것으로 관측됨. train 데이터의 True 개수가 4850개 임을 생각하면 위 idx에 편향되어 학습된다고 판단했고, test 셋에는 이 idx가 없는 것을 확인하여 위 2421개 중 일부를 샘플링하여 과적합을 줄이려는 시도
pubilc score 0.7

### 4. Voting
언더 샘플링 시 정보손실의 문제가 있음.
False 데이터 54449 개를 랜덤 셔플 후, 모두 20등분하고 True와 합쳐 클래스 비율이 1:1인 데이터셋 20개를 생성.
각각 데이터셋의 모델에서의 결과를 확률로 받은 후 0, 1 클래스의 확률을 평균을 내어 최종 결과로 생성 (Soft voting)
public score 0.02 정도 상승을 보임

### 진행했던 AB test
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/c13ca29e-1cb5-4f1a-a44a-db35272ee936" alt="data" width=70% height=70%>

모델 학습은 GridSearch를 이용


<br/>
## 5. 결과
<br/>

### 모델 선택
앞서 선택한 5개 모델 중 5개, 3개, 1개로 나누어 앙상블 후 가장 public score가 높은 모델 선택
-> 'xgb' 1개 사용시 가장 성능이 높음.

### 최종 결과
Public Score : 0.75611
Final Score : 0.76485

844팀 중 63위로 본선 진출 실패 (30위 팀 Final Score : 0.78086)
