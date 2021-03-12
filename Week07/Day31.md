# Week07 - CV, Computer Vision

## [Day 31] - 

### 1. Image Classification I

이번 강의에서는 computer vision의 소개와 더불어 가장 기본적인 과제인 Image Classification에 대해 다루게 됩니다. Image Classification은 주어진 사진을 특정 카테고리로 분류하는 과제입니다.

기존 머신러닝과 구분되는 딥러닝 기반 Image Classification의 특징에 대해서 배우고 대표적인 CNN 모델인 AlexNet과 대표적인 classification 모델인 VGGNet에 대해 배우게 됩니다.

#### 1) Computer Vision overview

computer vision은 AI의 한 분야로, AI가 추구하는 인간의 지능을 컴퓨터로 표현하는 목표 중 영상, 이미지와 같은 **시각**과 관련된 분야입니다. 

사람이 사물을 시각적으로 이해하는 과정에 대해서 간단하게 요약해볼 수 있습니다. 사물을 눈으로 관찰하게 되면, 수정체 뒤에 상이 맺히게 됩니다. 다음으로 뇌에서는 해당 자극을 전달하고 이해하여 해석하는 과정을 거치게 됩니다.

마찬가지고 해당 과정을 컴퓨터 프로세스에서 구현해볼 수 있습니다. 사물이 담긴 사진 혹은 영상이 있다면 구축한 모델과 환경에서 목적에 맞게 분석하여 결과가 출력되게 됩니다.

<image src = https://user-images.githubusercontent.com/48677363/110748509-28b1ce00-8283-11eb-9d5d-26e3db40fc57.png width = 700>

결국 Computer vision은 영상이나 이미지와 같은 visual data의 특징과 과제 목적에 맞게 이해하고 활용하게 하는 것이라고 할 수 있습니다. 시각 지각 능력에 대한 목적을 구체화하게 되면 다음과 같습니다.

- Color perception
- Motion perception
- 3D perception
- Sematic-level perception
- Social perception (emotion perception)
- Visuomotor perception

#### 2) Image classification

compuyter vision에서 image classification은 가장 단순하고 중요한 과제 중 하나입니다. 주어진 영상 혹은 이미지가 입력된다면 특정 카테고리로 분류하는 과정입니다. 

image classification의 가장 간단한 모델로는 모든 픽셀들에 weight를 부여하는 Fully-connected layer를 사용할 수 있습니다. 

<image src = https://user-images.githubusercontent.com/48677363/110890740-f2805700-8333-11eb-87bd-0082c849f9ac.png width = 700>

하지만 모든 픽셀에 weight를 매핑하게 되면 다양한 형태의 이미지의 특성을 모두 반영하기 어렵습니다. 또한 학습 데이터 이외의 패턴을 가지는 데이터를 곡해할 수 있는 가능성이 보다 높습니다. 이러한 단점을 보완하기 위해서 이미지의 특성을 국소적으로 학습할 수 있는(locally connected) Comnvolution Neural Network가 등장하게 되었습니다.

<image src = https://user-images.githubusercontent.com/48677363/110892118-cc0feb00-8336-11eb-928a-49746473a5c0.png width = 700>

영상과 이미지에서 효과적인 성능을 발휘하는 CNN은 다양한 Computer vision 과제에서 backbone network(base network)로 주로 사용되고 있습니다. CNN을 통해 데이터의 feature(특성) map을 도출 및 형성하고 과제 목적에 맞게 후속 layer를 모델링하는 것이 computer vision의 기본적인 설계 과정이라고 할 수 있습니다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/110892533-94ee0980-8337-11eb-8e0f-ca3822719ff3.png width = 500>
</center>

#### 3) CNN architectures for image classification I





---------

### 2. Annotation data efficient learning

딥러닝 기반 Computer Vision에서는 supervised learning의 학습 방법이 보다 높은 성능을 보장한다고 알려져 있습니다. 하지만, 딥러닝 모델을 학습하기 위한 대량의 고품질 데이터를 확보하는 것은 보통 불가능하거나 비용이 매우 큽니다.

이번 강의에서는 딥러닝 모델의 성능을 높이기 위한 다양한 기법에 대해서 배웁니다. Data Augmentation, Knowledge Distillation, Transfer learning, Learning without Forgetting, Semi-supervised learning 및 Self-training 등 주어진 데이터셋의 분포를 실제 데이터 분포와 최대한 유사하게 만들거나, 이미 학습된 정보를 이용해 새 데이터셋에 대해 보다 잘 학습하거나 label이 없는 데이터를 이용해 학습하는 등의 기법을 통해 효율적인 딥러닝 학습을 꾀할 수 있습니다.

