# Week03 - DL Basic

## [Day 12] - 최적화

### 1. Optimization

#### 1) Optimization의 기본 용어

  - **Generalization**

보통 딥러닝에서는 Generalization, 일반화 성능을 높이는 것을 목적으로 한다. train 데이터로 학습한 모델은 시간이 지나면 test 데이터에 대한 error가 커질 수 밖에 없다. generalization performance는 일반적으로 train 데이터 error와 test 데이터 error와의 차이를 의미한다.

그러므로 예상치 못한 test 데이터를 예측함에 있어서도 error가 크게 발생하지 않는 견고한 모델을 generalization이 우수하다고 볼 수 있다.

<image src = https://user-images.githubusercontent.com/48677363/106565612-6def2b80-6572-11eb-887c-36ab32ddefda.png width = 500>

  - **Underfitting vs Overfitting**

underfitting과 overfitting은 train 데이터에 대한 모델의 학습 정도를 의미한다. underfitting은 신경망의 깊이가 너무 얇거나, 학습 횟수가 적어 train 데이터 조차도 제대로 예측하지 못하는 경우를 의미한다. 반대로 overfitting은 모델이 train 데이터의 특징만을 고려하여 학습되어 test 데이터의 특징을 전혀 고려하지 못하는 경우를 의미한다.

<image src = https://user-images.githubusercontent.com/48677363/106566384-94619680-6573-11eb-9ddf-6374f01ddc02.png width = 550>

  - **Cross-validation**

보다 generalization한 모델을 구축하기 위해서 대부분의 경우에 본래 가지고 있던 train 데이터를 train, valid로 구분하여 학습을 진행하게 된다. 이렇게 validation 과정을 거침으로써 보다 최적화된 hyperparameters들을 찾을 수 있으며, 이 때 가장 중요한 것은 test 데이터는 학습 과정에서 어떠한 방법으로도 사용되어서는 안된다.

<image src = https://user-images.githubusercontent.com/48677363/106567270-eb1ba000-6574-11eb-97a4-1805d7d656e9.png width = 500>

  - **Bias and Variance**

모델의 에러를 최소화 하는 것은 bias와 variance의 trade-off를 의미한다.

<image src = https://user-images.githubusercontent.com/48677363/106567400-1b633e80-6575-11eb-9598-84abf38cb939.png width = 300>

<image src = https://user-images.githubusercontent.com/48677363/106569971-9c700500-6578-11eb-9498-2e36602d6bb9.png width = 450>

  - **Bootstrapping**

학습 데이터가 고정되어 있을 때, sub sampling을 통해 여러 개의 학습 데이터를 생성하고 여러 모델 및 평가지표를 적용해서 표본집단에서 보다 정확한 모수를 얻을 수 있다.

  - **Bagging vs Boosting**

**Bagginng(Boostrapping aggregating)** 은 sub sampling을 통한 다수의 학습 데이터에 다양한 모델을 적용하여 결과를 합쳐 하나의 새로운 결과를 출력하는 방법이다. 이 때, 하나의 새로운 결과로 취합하는 과정에서는 평균 혹은 voting 등의 방법이 사용된다.

**Boosting**은  학습 데이터에 각 모델들을 독립적으로 적용하여 독립적인 결과를 취합하는 것이 아닌, 전체 학습 데이터에 모델들을 상호 취합하여 결과를 출력하는 방법이다, 

<image src = https://user-images.githubusercontent.com/48677363/106571397-7186b080-657a-11eb-8a3e-4920a9c7d9af.png width = 500>

#### 2) Practical Gradient Descent Methods

  - Stochastic gradient descent: 데이터 1개씩(single sample) gradient를 구하면서 업데이트를 진행하는 방식

  - Mini-batch gradient descent: 전체 데이터 중 일부만을 가지고 gradient를 구해 업데이틑 진행하는 방식

  - Batch gradient descent: 전체 데이터를 모두 사용해 gradient를 구해 업데이트를 진행하는 방식

**Batch-size Matters**

hyperparameter 중 하나인 Batch size가 보통 64, 128, 256 등의 수치를 기본적으로 사용하는 편이지만 굉장히 중요한 hyperparameter이다.

large batch size를 사용하게 되면 sharp minimizers에 도달하게 되고
small batch size를 사용하게 되면 flat minimizers에 도달하게 된다. 

보통 small batch size를 적용하게 되면 test 데이터에 적용하게 되어도 보다 generalization된 결과를 얻을 가능성이 높다. (약간 learning rate와 비슷한 느낌을 받았음)

#### 3) Gradient Descent Methods

  - **Gradient Descent**

  - **Momentum**

  - **Nesterov Accelerated Gradient(NAG)**

  - **Adagrad**

  - **Adadelta**

  - **RMSprop**

  - **Adam**

#### 4) Regularization

  - **Early stopping**

  - **Parameter norm penalty**

  - **Data augmentation**

  - **Noise robustness**

  - **Label smoothing**

  - **Dropout**

  - **Batch Normalization**

---------

### 2. CNN 기초

Convolution 연산은 이미지 혹은 영상을 처리하기 위한 모델에서 굉장히 많이 사용된다. 이 전 강의에서 배웠던 MLP와 비교해서 CNN, Convolutional Neural Network의 커널 연산이 가지는 장점과 다양한 차원에서 진행되는 연산 과정에 대해서 배우게 된다.

Convolution 연산의 경우, 커널의 모든 입력 데이터에 대해 공통으로 적용이 되기 때문에 역전파를 계산하는 경우에도 똑같이 Convolution 연산이 출력된다. 강의에서 그림과 함께 잘 설명되어 있기 때문에 커널을 통해 gradient가 전달되는 과정과 역전파 과정에 대해 숙지해야한다.

#### 1) Convolution 연산

이 전 강의에서 배운 다층신경망, MLP는 각 뉴런들이 선형모델과 활성함수로 모두 연결된, fully connected 구조 였다. 다층신경망은 입력데이터와 가중치 행렬의 i 위치와의 행렬곱을 통해서 계산이 이뤄지기 때문에 가중치 행렬의 구조가 복잡하고 커지게 되어 학습 파라미터가 많아지게 된다.

<image src = https://user-images.githubusercontent.com/48677363/106689647-adba1f80-6613-11eb-9fd5-c76bbe91e290.png width = 500>
<center> [ fully connected 연산 ] </center>

<br>
<br>

반면에, **Convolution 연산** 은 커널, Kernel 이라는 고정된 가중치를 사용해 입력벡터 상에서 움직이며 선형모델과 합성함수가 적용되는 구조이다. 입력 데이터를 모두 동일하게 활용하는 것이 아니라 커널 사이즈에 대응되는 입력 데이터만을 추출하여 활용함을 의미한다.
활성화 함수를 제외한 Convolution 연산도 선형변환에 속한다. 

<image src = https://user-images.githubusercontent.com/48677363/106689976-4781cc80-6614-11eb-8f33-df1fcd78ab0f.png width = 500>
<center> [ convolution 연산 ] </center>

<br>
<br>

Convoltuion 연산의 수학적인 의미는 신호(signal)를 커널을 이용해 국소적으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것이다. 일반적으로 CNN에서 사용되는 Convolution 연산은 구체적으로 보자면 빼기가 사용되지 않고 더하기가 사용된 cross-correlation이다.
커널은 정의역 내에서 움직여도 변하지 않고(translation invariant) 주어진 신호에 국소적(local)으로 적용된다.

#### 2) 2D Convolution 연산

2D Convolution 연산은 커널을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조이다. 2차원 커널은 다음과 같이 주어져있을 때, 입력 데이터에는 커널의 크기만큼 element-wise 곱셈처럼 계산된다.

이 때, 입력 데이터에 커널이 이동하면서 계산이 이뤄지는데, 이동 범위에 해당하는 hyperparameter가 stride이다. 

<image src = https://user-images.githubusercontent.com/48677363/106712012-b45c8d00-663b-11eb-881b-b9bf2c6f5542.png width = 600>

커널의 크기와 입력 데이터의 차원을 미리 알고 있다면 계산되는 convolution 연산의 출력을 계산해볼 수 있다. 입력 데이터 크기를 (H, W), 커널 크기를($K_H$, $K_W$)라 한다면 출력 크기를 ($O_H$, $O_W$) 다음과 같이 계산할 수 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/106713499-e53dc180-663d-11eb-94cf-d2f2a35ac3f1.png width = 300>
</center>

즉, 채널이 여러개인 2차원 이상의 데이터의 경우에는 2차원 Convolution을 채널 개수만큼 적용하게 된다. 결국 입력 데이터의 채널의 개수만큼 커널이 적용되어 계산이 이뤄진다면 출력 데이터는 1개의 채널의 2차원 데이터가 될 것이다.

만약 출력 데이터 또한 채널의 개수 혹은 다차원이기를 원한다면 적용되는 커널의 차원 자체를 증가시켜 적용하게 되면 출력 데이터는 입력 데이터에 독립적으로 적용되는 커널의 개수만큼 나오게 된다.

<image src = https://user-images.githubusercontent.com/48677363/106714661-80836680-663f-11eb-94ae-30ce206cc7e7.png width = 600>

#### 3) Convolution 연산의 역전파

출력 데이터는 입력 데이터와 커널과의 element-wise 곱셉으로 계산된 데이터이다. 그렇기 때문에 역전파 또한 convolution 연산처럼 커널 크기에 따라 진행이 된다.

<image src = https://user-images.githubusercontent.com/48677363/106716437-c93c1f00-6641-11eb-841f-9bc4e8f7cc34.png width = 500>

---------

### Further Questions

#### 1) 올바르게(?) cross-validation을 하기 위해서는 어떻 방법들이 존재할까요?


#### 2) Time series의 경우 일반적인 k-fold cv를 사용해도 될까요?
