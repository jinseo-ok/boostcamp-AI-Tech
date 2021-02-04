# Week03 - DL Basic

## [Day 14] - Recurrent Neural Networks

### 1. RNN 기초

Sequence 데이터의 개념과 특징 그리고 이를 처리하기 위한 RNN, Recurrent Neural Networs에 대해서 다룬다.
RNN에서의 역전파 방법인 BPTT와 기울기 소실 문제에 대해 자세하게 다룬다.

시퀀스 데이터만이 가지는 특징과 종류, 처리 방법 그리고 이를 위한 RNN의 구조를 CNN 혹은 MLP와 비교하여 학습하는 것이 중요하다. 
RNN의 역전파 방법인 BPTT(Back Propagation Through Time)를 수식적으로 이해하고 해당 과정에서 발생하는 기울기 소실 문제와 이에 대한 해결책을 중심으로 RNN을 학습할 수 있다.

#### 1) Sequence data

**Sequence** 라는 영어 단어의 의미를 보면 '(일련의) 연속적인 사건들[행동들/숫자들 등]'이라는 것을 확인할 수 있다. 그렇다면 **Sequence data** 는 시간 순서에 따라 나열된 데이터를 시퀀스 데이터로 분류한다.

소리, 문자열과 언어, 주가 등이 시퀀스 데이터의 예로 볼 수 있다. 시퀀스 데이터는 독립동등분포(i.i.d) 가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀌게 된다.

#### 2) Sequence data 처리

이전 시퀀스의 정보를 가지고 앞으로 발생할 데이터의 확률분포를 다루기 위해 조건부확률을 이용할 수 있다. 

시퀀스 데이터를 다룰 때에는 베이즈 법칙을 통해서 X1부터 Xt까지의 결합분포를 Xt에 대한 조건부확률 분포와 X1부터 Xt-1까지 결합확률을 곱해주어 구할 수 있다. 이러한 조건부확률을 반복적으로 분해하게 된다면, X1부터 Xt-1까지의 데이터로 Xt를 추론하는 조건부확률의 곱으로 나타낼 수 있다.

<image src = https://user-images.githubusercontent.com/48677363/106879296-4e950180-671e-11eb-8fbc-f610a09ccec2.png width = 500>

즉, 시퀀스 데이터는 순서의 개념이 도입되기 때문에 이전 시퀀스의 정보라는 조건으로 앞으로 발생할 데이터의 확률분포를 다룰 수 있는데, 과거 정보를 무조건 모두 필요하거나 활용한다는 의미는 아니다. 또한 시퀀스 데이터를 다루기 위해서는 길이가 가변적인 데이터를 다룰 수 있는 모델이 필요하다.

고정된 길이 T만큼의 시퀀스만 사용하는 경우 AR(Autoregressive Model), 자기회귀모델이라고 부르며, 바로 이전 정보를 제외한 나머지 정보들을 다른 잠재변수로(Ht) 인코딩해서 활용하는 잠재 AR 모델이라 한다.

자귀회귀모델에서 Ht를 인코딩하는 방법을 해결하기 위해서 적용된 알고리즘이 RNN이다. Neural network를 통해서 직전 이전의 정보들을 인코딩하여 현재 정보를 예측하는 모델 구조를 보인다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/106881250-9cab0480-6720-11eb-82e8-aadf0b9bd4c4.png width = 400>
</center>

#### 3) Recurrent Neural Network

RNN에서 Ht라는 잠재변수를 인코딩하는 과정은 MLP의 layer에서 발생하는 계산 과정과 유사하다.

MLP에서는 입력 데이터 X는 가중치 행렬의 곱과 bias의 합으로 선형결합이 이뤄지고, 활성화 함수를 거쳐 잠재변수로 생성된다. 다음 layer에서는 해당 잠재변수에 반복적인 선형결합과 활성화 함수를 통해 최종 출력이 나오게 되는데, 곱해지는 가중치 행렬 W1과 W2는 시퀀스와 상관없이 불변인 행렬로 이해할 수 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/106881250-9cab0480-6720-11eb-82e8-aadf0b9bd4c4.png width = 400>
</center>

하지만 RNN에서는 과거의 정보를 Ht안에 담기 위해서 이 전 잠재변수(Ht-1)로 부터 정보를 전달받게 되는 새로운 가중치 행렬인 WH가 필요하게 된다. 즉, 이전 잠재변수가 다음 잠재변수를 생성하는 과정에서 결합되기 때문에 정보를 보존된 상태에서 새로운 출력을 할 수 있다는 개념이다.

RNN의 역전파는 잠재변수의 연결그래프에 따라 순차적으로 계산된다. 출력과 다음 시점에서의 잠재변수의 gradient vector이 입력되면서 학습이 이뤄진다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/106882508-2dceab00-6722-11eb-9d9a-e1501222272c.png width = 500>
</center>

#### 4) Back Propagation Through Time

BPTT를 통해 RNN의 가중치행렬의 미분을 계산해보면 아래와 같이 미분의 곱으로 이루어진 항이 계산된다. 최종적으로 나오게 되는 i시점부터 t시점까지의 모든 잠재변수에 대한 미분값의 곱은 시퀀스 길이가 길어질수록 불안정할 가능성이 높다.

항들의 값이 1보다 크게 되면 미분값이 매우 커지게 되고, 1보다 작게 되면 미분값이 매우 작게 된다. 그렇기 때문에 일반적인 BPTT를 모든 시점에 적용하게 되면 RNN의 학습이 불안정하게 된다. 특히 기울기 소실 문제인 gradient vanishing problem이 발생하게 되면 과거 시점에 대한 예측이 거의 불가하기 때문에 미분값의 범위를 정해 계산하는 truncated BPTT 방법이 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/106883681-a8e49100-6723-11eb-91f5-b1f79fdcf383.png width = 300>
</center>

하지만 truncated BPTT 또한 gradient vanishing을 무조건적으로 해결할 수 있는 것은 아니기 때문에 LSTM, GRU 등과 같은 새로운 개념들이 제안되기도 했다.

---------

### 2. Sequential Models - RNN

Sequential data는 시작과 끝이라는 시점의 특징을 가지고 있는 주식 혹은 언어 데이터에 해당한다. 이러한 Sequential data에 적용하기 위한 Sequential model의 정의와 종류에 대해 배운다.
그 후, 딥러닝에서 sequential data를 다루는 RNN에 대한 정의와 종류에 대해 배운다.




---------


### 3. Sequential Models - Transformer

Sequential model의 한계점과 이를 해결하기 위해 등장한 Transformer에 대해 배운다. 특히 Transformer의 구조 중 Encoder와 Multi Head Attention의 구조와 개념에 대해 보다 집중적으로 배운다.


----------

