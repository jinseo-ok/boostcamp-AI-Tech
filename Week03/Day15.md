# Week03 - DL Basic

## [Day 15] - Generative Models

### 1. Generative Models I

#### 1) Generative Models 개념

**Generative, 생성** 은 '사물이 생겨남. 또는 사물이 생겨 이루어지게 함.'을 의미한다. Generative Models, 생성 모델이 무언가를 만들어내는 모델로 이해할 수 있지만 생성 모델이 가지는 의미에 따라 다양한 과제를 수행할 수 있다.

  - Generation: 학습 후 해당 학습 데이터와 유사한 이미지를 만들어내는 목표 
  - Density estimation: 학습 후 이미지 분류를 하는 목표, anomaly detection에 활용될 수 있음
  - Unsupervies representation learning: 이미지의 보편적인 특징을 학습하는 목표

Density estimation를 보면 분류와 같은 Discriminative의 특징을 가지고 있다. 그렇기 때문에 생성 모델은 단순히 무언가를 만들어내는 모델에 국한되는 것이 아닌 생성과 구분 등과 같은 범위로 확대할 수 있다.

이렇게 입력이 주어졌을 때, 확률과 같은 결과가 출력되는 모델은 explicit generative model이라고 하고 생성 결과를 출력하는 모델을 implicit generative model이라고 한다.

#### 2) Generative Models 알고리즘

생성 모델이 입력되는 데이터의 분포를 파악하기 위해서는 기본적인 확률 분포에 대한 지식이 필요하다.

  - Bernoulli distribution
    - 확률이 P, 1-P인 coin flip
    - 2가지 경우에 대한 확률 표현은 P 1개만 필요
<br>
  - Categorical distribution
    - N개의 경우에 대한 확률 표현은 N-1개 필요

해당 분포에 의거해서 RGB 픽셀을 표현한다면 256가지의 경우가 존재하는 하나의 채널에서 categorical distribution에 의해 255 x 255 x 255개의 parameter가 계산된다. 또한 2가지 경우만 존재하는 흑백 이미지를 표현하기 위해서도 $2^{n} - 1$개의 parameter가 필요하게 된다. 여기서 발생하는 문제는 parameter가 방대하게 되면 모델이 데이터를 학습하는데 어려움을 가진다는 것이다.

학습해야하는 parameter의 개수가 너무 많은 것을 줄이기 위해서, N개의 픽셀을 모두 독립적이라고 가정하는 방법이 있다. 모든 픽셀이 독립적이라고 가정하게 된다면 하나의 픽셀을 표현하고자 하는 parameter 자체는 1개로 표현가능하기 때문에 전체 parameter는 N개가 된다. 

하지만 이렇게 모든 픽셀을 독립적이라고 가정하게 된다면 표현할 수 있는 범위가 굉장히 작아지기 때문에 모든 픽셀을 표현하고자하는 방법과 독립적이라고 가정한 방법 사이의 협의점을 찾아야 한다. 이 때, 3가지 테크닉이 적용된다.

Chain rule과 Bayes' rule은 항상 만족하는 조건이며 Conditional independece는 가정을 의미한다. Conditional independence는 특히 chain rule과 활용하면서 fully dependent한 생성 모델에 가까운 확률 분포를 출력하고자 한다.

  - Chain rule

<center>
<image src = https://user-images.githubusercontent.com/48677363/107754343-b1148000-6d64-11eb-887d-f960658f67be.png width = 400>
</center>

  - Bayes' rule

<center>
<image src = https://user-images.githubusercontent.com/48677363/107754417-c9849a80-6d64-11eb-9511-1ddccbc79f62.png width = 200>
</center>

  - Conditional independece: z가 주어졌을 때, x와 y는 독립적이다. z가 주어졌을 때, x를 표현함에 있어 y는 상관없다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107754510-ed47e080-6d64-11eb-8ecd-c12bcdab9d2e.png width = 250>
</center>

**Conditional independece**

Conditional independece에는 Markov assumption이 적용된다. N번째 데이터는 N-1 데이터에만 dependent함을 가정하는 Markov assumption을 chain rule로 얻어지는 조건부확률의 곱이 자연스럽게 생략된다.

위에서 다뤘던 Chain rule의 수식과 Markov assumption이 적용된 수식을 보면 차이를 바로 알 수 있다. 즉, 이렇게 식을 전개했을 때 parameter를 계산하게 되면 **2n - 1** 개가 필요하게 된다. 결론적으로 $2^{n}$ 혹은 $2^{n} - 1$개의 parameter가 필요했던 모델에서 매우 적은 parameter로 학습이 가능하게 된다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107758207-e96a8d00-6d69-11eb-99d5-d2f76548ea46.png width = 400>
</center>

#### 3) Auto-regressive Model

Conditional independece를 활용한 모델이 Auto-regressive Model이다. 좀 더 명확하게 짚고 넘어가야할 점은 Auto-regressive Model은 t 시점의 데이터가 t 시점 이전의 데이터에 dependent하다는 것을 의미한다. 그렇기 때문에 markov assumption이 적용된 t 시점 데이터가 t-1 시점 데이터에만 dependent한 모델도, 그리고 t 시점 이전 모든 데이터에 dependent한 모델도 Auto-regressive model인 것이다.

Auto-regressive Model를 활용하기 위해서는 데이터의 ordering이 중요하다. 하지만 이미지의 경우에는 픽셀 데이터의 순서를 명확하게 ordering 하는 것에 어려움이 있을 수 있다. 이 때는 행순, 열순으로 임의로 규칙을 부여하여 ordering 할 수 있다.

또한 고려해야하는 시점의 데이터에 따라서 Conditional independece가 달라지고 전체 모델의 구조가 달라지게 된다.

**NADE: Nerual Autogressive Density Estimator**

<center>
<image src = https://user-images.githubusercontent.com/48677363/107841970-01421f80-6e03-11eb-9665-695659eca5da.png width = 500>
</center>

  - i 번째 데이터를 1 ~ i-1 번쨰 데이터에 dependent함을 가정한 모델
  - i 번째 픽셀은 i-1 개의 입력이 발생하게 됨
  - ordering에 따라 입력 차원이 계속 증가하게 됨
  - generation뿐만 아니라 확률 계산이 가능하기 때문에 **explicit model**

--------

### 2. Generative Models II

#### 1) VAE, Variational Autoo-encoder

##### (1) Variational inference(VI)

  - VI의 목표는 posterior distribution과 일치?하는 variational distribution을 최적화하는 것
  - posterior distribution이란 관측 데이터에서 관심있는 random variables의 확률분포를 의미함
  - variational distribution이란 일반적으로 posterior distribution를 찾기 굉장히 어렵기 때문에 모델을 통해 근사한 확률분포를 의미함
  - KL divergence 평가지표를 사용하여 posterior distribution과 variational distribution의 loss를 최적화함

<image src = https://user-images.githubusercontent.com/48677363/107843824-892f2600-6e11-11eb-811a-e95b87a1ffbc.png width = 500>

사실 posterior distribution의 true를 알지 못하게 되면 loss를 줄여나가면서 variational distribution를 근사하려는 방법은 어불성설이다. 해당 과정을 가능하게 해주는 것이 Variational Inference의 **ELBO**이다.

**ELBO**

<image src = https://user-images.githubusercontent.com/48677363/107843948-d069e680-6e12-11eb-844d-d9fd37a0dbfa.png width = 500>


#### 2) GAN, Generative Adversarial Network

