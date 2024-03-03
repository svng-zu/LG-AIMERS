# 손목 뼈 X-ray 사진을 통한 골연령 예측 프로젝트

## 1. 개요
이미지 분석을 통해 골 연령을 산출하여 성장 과정 동안의 키 성장을 예측 <br/>
이번 프로젝트에서는 골연령을 예측하는 것이 목표 <br/>

## 2. 이미지 전처리
### Resize
각 이미지마다 크기가 다르기 때문에 이미지 크기를 동일하게 설정
```
# 전체 이미지의 비율 평균 계산하여 resize 할 비율 결정
ratio = 0
for i in range(len(total_df.id)):
    img = cv2.imread(total_path + total_df['id'][i], cv2.IMREAD_GRAYSCALE)
    ratio += img.shape[0] / img.shape[1]

ratio / len(total_df.id) ## 1.266
```
```
# 이미지 사이즈 재설정
img = cv2.imread(total_path + '98.jpg', cv2.IMREAD_GRAYSCALE)
resize_img = cv2.resize(img, (800, 1000))
```
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/95376baf-2b8d-4c23-9bf2-8261a0efcc51)

### Normalize
```
# 이미지 min max 정규화
normal_img = cv2.normalize(resize_img, None, 0, 255, cv2.NORM_MINMAX)
```
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/fa0b4da2-f7a0-4466-be66-9809df4bd374)

### Equalize
createCLAHE(Contrast Limited Adaptive Histogram Equalization) <br/>
이미지를 여러 작은 블록으로 나누어 각 블록에 대해 독립적으로 히스토그램 평탄화를 적용
```
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(3,3))
equal_img=clahe.apply(denoise_img)
```
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/25ca3ff7-3151-4ed8-8a7a-3f65aa7bf4d7)

## 3. 이미지 회전
### 샘플 이미지
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/0af5cdd7-f61c-49df-a529-5d3b1956d650)

### Thresholding
0(검은색), 255(흰색)로 이진화
```
r_img = np.copy(temp_img)
height, width = temp_img.shape
img = temp_img[0:(int)(height*0.9),0:(int)(width*0.95)]

# 픽셀 값의 평균 크면 255로 설정되고 작으면 0으로 설정정
ret, img = cv2.threshold(img, temp_img.mean(), 255, cv2.THRESH_BINARY)

#샘플 이미지 Thresholding 결과
plt.imshow(img,"gray")
```
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/6feb6966-19ea-4bb8-b05a-9d20e7af4afa)

### 이미지 contouring
객체의 윤곽선 추출
```
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#외곽선 검출하여 mask 그리기
max_cnt = max(contours, key=cv2.contourArea)

mask = np.zeros(img.shape, dtype=np.uint8)
cv2.drawContours(mask, [max_cnt], -1, (255,255,255), -1)

plt.imshow(mask, 'gray')
```
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/26f14b7a-e017-4b0c-89c8-97943d218f5e)

### 첫 번째 흰색 좌표 구하기
y좌표 기준으로 첫 번째 흰색 좌표가 중지 끝부분임
```
M = cv2.moments(max_cnt)
center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

for y,x_r in enumerate(mask) :
    if 255 in x_r:
        #y에 따른 x rows 중 255인 x값 추출
        x_255_indexs = np.where(x_r == 255)[0]

        #255인 x값들 중 median 추출
        x_255_mid_index = x_255_indexs[len(x_255_indexs)//2]
        first_255_x_point = x_255_mid_index

        first_255_y_point = y
        break

(first_255_x_point,first_255_y_point) # (298, 102)
```

### 무게중심과 좌표의 각도 구하여 회전
```
# 이미지 회전 함수 정의
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    return result
```
```
# 무게중심과 첫 흰색 좌표 차이
center_x, center_y = center[0], center[1]
rx = abs(first_255_x_point - center_x)
ry = center_y - first_255_y_point

# 회전 각도 구하기
import math
radian = math.atan2(ry, rx)
degree = 90 - math.degrees(radian)

# 무게중심과 첫 좌표 위치에 따라 회전 방향 조정

if first_255_x_point < center_x :
    mask = rotate_image(mask,360-degree) 
    r_img_ = rotate_image(temp_img,360-degree)
else:
    mask = rotate_image(mask,degree) 
    r_img_ = rotate_image(temp_img,degree) 
```
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/1c752d35-6ff8-4b10-9bc1-3a8aa5d61c1a)

## 4. ROI 추출
cv2.convexHull, cv2.convexityDefects 를 이용해 오목한 부분의 좌표를 구하여 좌료로써 ROI 추출을 진행한다. <br/>
자세한 부분은 image_roi.ipynb 참고<br/>
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/1ce27c79-9c0d-4458-9543-59f5f459562c)

### 손목뼈 좌표 추출
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/2042af7c-9b20-4839-a7d2-afcae5e9bad3)

### 손목뼈 위쪽 관절 추출
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/608ae174-84a7-46f0-86f7-65396aeb993c)

### 중지 좌표 추출
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/ce941278-6fff-42cd-a5e5-05e3a1c909b7)

### 엄지손가락 좌표 추출
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/bb567181-80df-48bd-8a16-ab6fef7c55e1)

### cropping
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/82c4701c-ac36-45e1-aef1-0dadcb9727a4)

## 5. data Augmentation
현재 데이터가 1234개로 학습하기에는 데이터가 부족함 <br/>
데이터 증강을 통해 학습시의 과적합을 방지하고 좀 더 유연한 모델을 만들어본다. <br/>

### Rotation
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/cb9a65cc-397e-42cf-8e91-d73e7269d58d)

### Add noise
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/f631b3c8-384c-4c15-8545-09721c6bfe2e)

### 최종 학습 데이터의 형태
![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/541745f8-75d6-4180-b0ea-552c8617f893)

## 6. CNN 모델 생성 - Attention-Xception
아래 논문 참고하여 모델을 생성하였음 <br/>
[Intelligent Bone Age Assessment: An Automated System to Detect a Bone Growth Problem Using Convolutional Neural Networks with Attention Mechanism](https://www.researchgate.net/publication/351221264_Intelligent_Bone_Age_Assessment_An_Automated_System_to_Detect_a_Bone_Growth_Problem_Using_Convolutional_Neural_Networks_with_Attention_Mechanism)

<img src = 'https://github.com/Junoflows/BoneAge_Project/assets/108385417/090a9dbb-0c43-4d37-9bbc-fa95548cba10' width = 500 height = 400>
<img src = 'https://github.com/Junoflows/BoneAge_Project/assets/108385417/cffc654d-374c-49cd-af3b-cefa0cbabcf9' width = 500 height = 400>

위 사진은 사용할 모델의 구성으로 input 으로 4차원 tensor, output 값은 1개이다. <br/>
loss 는 mae로 회귀로 학습한다.

## 7. Result
crop 이미지를 thresholding 여부로 실험 -> 이진화처리를 하지 않는게 더 정확도가 높음 <br/>
배치사이즈는 본인의 메모리에 따라 다르게 진행하는데 작을수록 더 정확했음 <br/>
데이터 증강하여 학습 데이터에 추가하였더니 정확도가 1.11 에서 0.78까지 상승 <br/>

### 최종 결과
최종적으로 mae : 0.785 으로 약 9.43 months 오차발생 <br/>

![image](https://github.com/Junoflows/BoneAge_Project/assets/108385417/00321a9e-dd2c-4e16-834d-e8db57b6d05d)
앞선 논문에 대한 실험 결과인데 논문 결과로는 약 7.7 months 오차가 발생하였다. <br/>

위 논문에서는 TW3 방식으로 관절의 ROI 추출과 정제된 X-ray 데이터 14235개의 데이터로 진행하였는데,
이 프로젝트는 TW3 와 같은 의학 부분의 전문적인 지식없이 ROI 추출하였고, 데이터의 개수도 1236개로 진행하였음 <br/>

그 부분을 감안하였을 때 9.43months 의 오차는 허용범위라고 판단하여 프로젝트를 종료하기로 결정



