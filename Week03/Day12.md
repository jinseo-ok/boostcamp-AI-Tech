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



---------

Further Questions

올바르게(?) cross-validation을 하기 위해서는 어떻 방법들이 존재할까요? 
Time series의 경우 일반적인 k-fold cv를 사용해도 될까요?
TimeseriesCV
Further reading

RAdam github
AdamP github