# Week03 - DL Basic

## [Day 13] - Convolutional Neural Networks

### 1. CNN

앞서 Day 12 에서 이미지 처리를 위해 효과적인 Convolutional Neural Networks, CNN의 기본적인 개념 대해서 간단히 알아보았습니다. 이번 시간에는 CNN 연산 방법과 연산에서 작용되는 hyperparameter가 기능들에 대해서 알아보겠습니다.

#### 1) stride

CNN 연산은 커널이라는 고정된 범위에 따라서 입력 데이터를 움직이며 연산이 이뤄지는 구조라고 언급했습니다. 여기서, 커널을 입력 데이터 상에서 움직이는 범위 혹은 간격을 결정하는 hyperparameter가 **stride** 입니다.

즉, stride 의 설정에 따라 고정된 범위의 커널이 받아들이는 데이터 중복의 범위가 결정된다고 할 수 있습니다. 보통 stride 가 작을수록 (1일 때..) 데이터를 학습함에 있어 장점을 가진다고 하는데, 아마 데이터 스킵없이 더 많은 데이터를 마주칠 수 있지 않기 때문이라고 생각합니다. 추가적으로 stride 는 하나의 값 뿐만 아니라 세로와 가로 모두 개별적으로 지정해줄 수 있습니다.

#### 2) padding

딥러닝에서 패딩, padding 이라는 개념은 전반적으로 비슷한 의미를 가지고 있습니다. 패딩이란 입력 데이터의 크기를 임의로 조정해주기 위해 특정 값을 사용하여 데이터의 크기를 맞춰주는 개념이라고 이해할 수 있습니다. CNN 에서는 커널이 입력 데이터를 이동하면서 데이터를 학습할 때, 가장자리 부분의 경우에는 학습하게 되는 횟수가 중앙 부분보다 적기 때문에 이를 방지하기 위해 입력 데이터의 크기를 임의로 늘려 데이터 손실을 방지하는 기능입니다. 이 때, padding에 사용되는 임의의 값으로는 보통 0인 zero-padding을 통상적으로 사용합니다.

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

<image src = https://user-images.githubusercontent.com/48677363/106866122-6dd86280-670f-11eb-9d06-ba674fbebf34.png width = 600>

##### (1) 기본 구조

  - skip connection이 포함되어 있는 residual connection를 추가함
  - network의 출력값과 입력값의 차이를 학습하기 위함
  - Simple shortcut과 Projected shortcut 중Simple shortcut 구조가 일반적으로 사용됨
  - Conv layer 이후에 bach norm, 이후에 activation 구조
  - 1 by 1 convolution을 통해서 채널의 차원을 조정함

##### (2) 핵심 아이디어

**skip connection**

<image src = https://user-images.githubusercontent.com/48677363/106866360-b6901b80-670f-11eb-957e-32f21dfc45ad.png width = 300>

#### 4) DenseNet(2017)

<image src = https://user-images.githubusercontent.com/48677363/106866476-d8899e00-670f-11eb-95fa-96a8a71716f7.png width = 500>


##### (1) 기본 구조

  - 이 전 layer에서 출력된 값을 다음 layer의 인풋으로 모두 활용하기 위해 단순 concat하여 사용하는 개념
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