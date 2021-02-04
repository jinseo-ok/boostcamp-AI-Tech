# Week03 - DL Basic

## [Day 13] - Convolutional Neural Networs

### 1. CNN

#### 1) Convolution 연산

#### 2) stride

#### 3) padding

<img src = https://i.stack.imgur.com/0rs9l.gif width = 500>

---------

### 2. Modern CNN

이번 강의는 ILSVRCF라는 Visual Recognition Challenge와 주요 대회에서 수상을 했던 5개 Network의 주요 아이디어와 구조에 대해 다룬다.

  - AlexNet
  - VGGNet
  - GoogLeNet
  - ResNet
  - DenseNet

#### 1) AlexNet(2012)

<image src = https://user-images.githubusercontent.com/48677363/106761840-8fd2d600-6678-11eb-9535-b8b72798fa3f.png width = 600>

##### (1) 기본 구조

  - 네트워크가 2개로 나누어져 있음(당시에는 부족한 GPU를 최대한 활용하기 위한 전략 때문)
  - 11 x 11의 filter를 사용함, paramter 관점에서 11 by 11를 사용하는 것은 효율적이지는 않음
  - 5층의 convolutional layers와 3층의 dense layers로 구성되어 있음

##### (2) 핵심 아이디어

2021년 현재에는 다음 아이디어가 굉장히 당연한 테크닉이지만 2012년도에는 혁신적인 기능들이었다.

  - ReLU activation function을 사용 -> gradient가 사라지는 문제를 해결
  - 2개의 GPU를 사용
  - Local response normalization(자세한 설명 생략, 현재 자주 사용하지 않는 기능이라 언급), Overlapping pooling 사용
  - Data augmentation 활용
  - Dropout 적용

#### 2) VGGNet(2014)

<image src = https://user-images.githubusercontent.com/48677363/106763026-c2c99980-6679-11eb-80b8-35bdb678690c.png width = 600>

##### (1) 기본 구조

  - 3 x 3의 filter를 사용
  - Dropout(0.5)를 사용
  - layer 16, 19층에 따라 VGG16, VGG19

##### (2) 핵심 아이디어

**3 x 3 filter**

filter의 크기에 따라 고려되는 인풋의 크기가 정해진다(Receptive field). 3x3의 filter를 2개의 layer로 네트워크를 구성하는 것과 5x5의 filter를 사용하는 것은 Receptive Filed 관점에서 동일하다. 하지만 parameter의 개수를 비교하면 다음과 같다.

  - 3x3: 3x3x128x128 + 3x3x128x128 = 294,912
  - 5x5: 5x5x128x128 = 409,600

같은 receptive field 관점에서 작은 사이즈의 filter를 사용함으로써 parameter의 개수를 줄일 수 있기 때문에 최근 논문의 모델에서는 오히려 작은 사이즈의 filter를 사용하는 경우가 많다.

#### 3) GoogLeNet(2015)

<image src = https://user-images.githubusercontent.com/48677363/106773704-b6970980-6684-11eb-97c5-fa92be0dbcca.png width = 600>

##### (1) 기본 구조

  - 총 22층의 layers로 구성
  - 비슷하게 보이는 network가 내부에서 여러번 반복됨, network in network 구조(NiN 구조)
  - Inception block을 활용

##### (2) 핵심 아이디어

**Inception block**

<image src = https://user-images.githubusercontent.com/48677363/106775552-95cfb380-6686-11eb-882c-9197493fcc0f.png width = 500>

Inception block은 인풋 데이터가 들어오게 되면 여러 갈래로 데이터가 퍼진 이후에 다시 concat되는 network 구조를 가진다. 각각의 경로를 보게 되면 모두 1x1 Conv layer를 거치는 것을 확인할 수 있다.

Inception blocok은 하나의 데이터로 여러 reponse를 모두 사용한다는 장점도 있지만 1x1 Conv를 통해서 parameter를 줄일 수 있는 문제가 보다 핵심적이다. 1x1 Conv는 채널방향으로 차원을 줄일 수 있기 때문이다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/106777830-b00a9100-6688-11eb-9748-748999de253b.png width = 500>
</center>

  - 3x3: 3x3x128x128 = 147.456
  - 3x3 -> 1x1: 1x1x128x32 + 3x3x32x128 = 4,096 + 36,864 = 40,960

#### 3) ResNet(2015)

##### (1) 기본 구조

  - skip connection이 포함되어 있는 residual connection를 추가함
  - network의 출력값과 입력값의 차이를 학습하기 위함
  - Simple shortcut과 Projected shortcut 중Simple shortcut 구조가 일반적으로 사용됨
  - Conv layer 이후에 bach norm, 이후에 activation 구조
  - 1 by 1 convolution을 통해서 채널의 차원을 조정함

##### (2) 핵심 아이디어

**skip connection**

#### 4) DenseNet(2017)

##### (1) 기본 구조

  - residual connection에서 더해지는 인풋 데이터를 단순 concat해서 사용하는 개념
  - layer 계속되는 데이터 차원이 기하급수적으로 증가
  - Transition Block을 통해 차원 사이즈를 줄여주게 됨(bach norm -> 1x1 Conv -> 2x2 AvgPooling)

---------

### 3. Computer Vision Application

Computer Vision 에서 CNN이 활용된 분야에 대해서 배운다.

Semantic segmentation의 정의, 핵심 아이디어에 대해, 그리고 Object detection의 정의, 핵심 아이디어를 통해 CNN이 어떻게 활용되는지 보다 구체적으로 알 수 있다.

#### 1) Semantic Segmentation

이미지 전체를 하나의 라벨로 분류하는 것이 아닌 이미지 안에 포함된 대상의 라벨을 모두 분류하는 문제이다.


---------



### Further Question

수업에서 다룬 modern CNN network의 일부는, Pytorch 라이브러리 내에서 pre-trained 모델로 지원합니다. pytorch를 통해 어떻게 불러올 수 있을까요?