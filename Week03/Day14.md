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

#### 1) Sequential Model

sequential model의 가장 기본적인 목적은 입력이 순차적으러 들어왔을 때, 다음 입력이 무엇일지 예측하는 것이다. 해당 모델의 가장 큰 특징으로는 특정 시점의 예측을 위한 조건이 시점에 따라 매번 달라진다는 것이다.

  - Autoregressive model: 모든 시점의 조건으로 과거 시점의 데이터의 개수를 정하는 모델
  - Markov model: 특정 시점의 이전 시점의 정보만을 활용하는 모델
  - Latent autogressive model: 과거의 정보를 함축하여 잠재변수를 생성하여 정보를 활용하는 모델

#### 2) Recurrent Neural Network 개요

RNN의 기본적인 구조는 MLP와 거의 유사하다. 하지만 위에서 살펴본 Sequential data를 다루기 위해서 자기 자신의 정보를 한번 더 참고하는 Recurrent(재순환) 구조가 추가되었다고 볼 수 있다.

재순환 구조가 적용된 모델을 시계열적으로 풀게되면 다음과 같은 형태의 구조를 가지게 된다. 해당 구조는 단순하게 바라보면 사실상 입력이 굉장히 많은 fully connected layer로 표현된 것이라 볼 수 있다. 

<image src = https://user-images.githubusercontent.com/48677363/107017481-85345000-67e2-11eb-9548-68c24b728221.png width = 600>

#### 3) Long-term dependencies

RNN의 가장 대표적인 단점으로 long-term dependecy라는 개념이 소개된다. 처음이라는 시점과 마지막이라는 시점이 존재하는 시퀀스 데이터에서는 특정 시점에서 그 시점 이전의 데이터를 모두 활용하는 것이 좋다.

하지만 시퀀스 데이터의 크기가 크게 되면, 후반 시점과 초기 시점의 간격이 차이나게 되면서 초기 시점의 정보를 잘 활용할 수 없게 되는 것이 long-term dependecy problem이라고 할 수 있다. 

<image src = https://user-images.githubusercontent.com/48677363/107042330-f7b62780-6804-11eb-98a1-7f968dc08fe7.png width = 600>

#### 4) LSTM, Long Short Term Memory

RNN 학습되는 과정은 다음과 같습니다. 모든 시점은 이 전 시점의 잠재변수로 생성된 Ht를 활용해서 다시 새로운 잠재변수를 생성하게 된다. 이런식으로 계산된 잠재변수가 계속해서 중첩되는 구조가 되면서 가중치를 곱하고 활성화 함수를 통과하게 되면서 정보의 가치를 잃어버릴 수 있다.

이 때, 활성화 함수에 따라서 정보가 너무 압축되거나 폭발하는 vanishing / exploding gradient 문제가 발생하게 된다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/107054205-2a671c80-6813-11eb-915e-44e1248fdbe4.png width = 600>
</center>

기본 RNN의 위와 같은 문제를 해결하기 위해서 다양한 컴포넌트를 추가함으로써 Long-term dependecy를 어느정도 해결할 수 있었다. LSTM의 전체적인 구조는 RNN의 일반적인 구조와 동일하지만 입력과 출력이 이뤄지는 과정에서 다양한 컴포넌트가 해당 문제를 해결하기 위해 움직인다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107057973-5dabaa80-6817-11eb-8fec-b51530cb054f.png width = 600>
</center>

특정 시점에서 입력된 데이터가 출력되고 다음 시점으로 전달되는 과정을 보다 구체적으로 보면 다음과 같다. 해당 모델을 언어 모델로 가정한다면 t시점에서의 입력 데이터는 단어(토큰)이 될 가능성이 높다. 보통 언어 모델에서는 해당 입력 데이터는 임베딩 데이터가 입력되게 된다.

출력 데이터는 잠재변수, hidden state이며 3개의 gate를 통해서 다음 입력에서 이 전의 정보를 활용하고자 한다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/107059171-ce06fb80-6818-11eb-9e94-7058201447d2.png width = 500>
</center>

  - **Forget Gate**: Decide which information to throw away
    - 어떤 정보를 버릴지 선택하는 구간
    - 이전의 hidden state와 새로운 입력인 x가 forget gate로 들어가게 됨
    - 시그모이드를 통해서 항상 0~1 사이의 값만을 가지게 되며 cell state 정보와 결합하게 됨

<center>
<image src = https://user-images.githubusercontent.com/48677363/107110282-05fa5700-688a-11eb-96c8-83a1e6baa775.png width = 600>
</center>

  - **Input Gate**: Decide which information to store in the cell state
    - 어떤 정보를 살릴지 선택하는 구간
    - 이전 hidden state와 입력 x를 결합하여 각각 $i_{t}$와 $C_{t}$를 형성함
    - 각각 생성된 두 개의 값을 결합하여 어떠한 정보를 업데이트할지 정하게 됨

<center>
<image src = https://user-images.githubusercontent.com/48677363/107110301-22968f00-688a-11eb-965d-334d6e407c0c.png width = 600>
</center>

  - **Update cell**: Update the cell state
    - forget gate의 결과와 이전 hidden state와의 결합과 input gate의 결과를 더해 새로운 cell state를 업데이트함

<center>
<image src = https://user-images.githubusercontent.com/48677363/107110463-64740500-688b-11eb-81fa-a6fd7174a3ce.png width = 600>
</center>

  - **Output Gate**: Make output using the updated cell state
    - 마지막으로 새롭게 생성된 cell state 정보와 output gate의 정보를 결합함으로써 해당 시점의 최종 hidden state를 출력하게 됨
    - Output gate를 생략한 과정이 GRU임

<center>
<image src = https://user-images.githubusercontent.com/48677363/107110479-88374b00-688b-11eb-9cbf-dfc3a852b0cb.png width = 600>
</center>

---------

### 3. Sequential Models - Transformer

Sequential model의 한계점과 이를 해결하기 위해 등장한 Transformer에 대해 배운다. 특히 Transformer의 구조 중 Encoder와 Multi Head Attention의 구조와 개념에 대해 보다 집중적으로 배운다.

#### 1) Transformer 개요

Sequential data는 현실적인 관점에서 순서라는 개념이 존재함으로써 생략, 뒤바뀜 등의 특징을 가지고 있으며 이러한 특징을 고려해서 sequential modeling을 하기에는 어려움이 존재한다. 예를 들어, 의사소통함에 있어 사람들은 백퍼센트 문법을 지키지 않는 경우가 대부분이며 대명사 혹은 문맥에 의한 생략이 정말 많다. 모델은 데이터 그 자체를 보기 때문에 이러한 특징을 보이는 sequential data를 모델링하기에 한계점이 존재한다.

RNN 혹은 LSTM과 같은 모델로는 이러한 문제를 해결하는 것에 어려움을 보였으며 Transformer는 이러한 문제를 보다 해결할 수 있는 구조를 가지고 있다. Transformer가 처음 소개 된 'Attention is All You Need(NIPS, 2017)' 논문에서 transformer에 대한 첫 소개다 바로 다음과 같다.

**'Tranformer is the first sequence transduction model based entirely on attention'**

transformer 구조는 RNN의 구조인 이 전 정보가 반복적으로 혹은 재귀적으로 입력되는 과정이 아닌 attention 매커니즘을 활용하여 sequential data를 다룬 새로운 구조를 선보인다.

해당 논문은 transformer 구조를 기계번역 문제에 초점을 맞추고 있지만 transformer는 sequential data를 처리하고 인코딩하는 방법으로 기계번역뿐만 아니라 이미지 classification, detection, visual transformer 등의 분야에서 활용되고 있다. 

#### 2) Transformer 구조

transformer는 기본적으로 encoder로 입력되는 문장을 decoder로 출력되는 문장으로 바꾸는 기계번역 알고리즘에 해당한다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/107193993-d6d31980-6a32-11eb-9c7f-a8a9b6fc0136.png width = 400>
</center>

transformer는 RNN 알고리즘처럼 입력 데이터 N번의 재귀적 순환이 존재하는 것이 아닌 한번에 입력 데이터 N개를 인코딩하는 과정인 self-attention 구조를 거친다. transformer에 대한 구조를 보다 정확하게 이해하기 위해서는 구조 내부에서 발생하는 3가지 과정에 대한 이해가 필요하다.

  - N개의 데이터가 순환 구조를 가지지 않고 인코딩 되는 과정에 대한 이해
  - encoder와 decoder간 어떤 상호작용이 발생하는 지에 대한 이해
  - decoder가 generation 하는 과정에 대한 이해

먼저, 입력 데이터로 3개의 단어가 self-attention layer를 통과하게 된다. self-attention layer를 통과하게 되면 각 단어(토큰)은 입력된 벡터에 따라 매핑되는 벡터가 출력되는데, 이 때의 벡터는 단순 MLP의 결과라기 보다는, 입력 데이터 간의 정보가 반영된 벡터라고 볼 수 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107195567-db98cd00-6a34-11eb-9f07-c8f8a33dd740.png width = 600>
</center>

#### 3) self-attention

그렇다면 어떤 과정으로 입력 데이터 간의 정보가 반영된 벡터가 매핑되는지 알아볼 수 있다. 다음과 같은 문장이 예시로 주어졌을 때, 컴퓨터가 ***it*** 단어를 그 자체로 인식하기 보다는 문장에서 다른 단어와의 관계 정보 포함된 의미를 파악하는 것이 보다 효과적이다.

self-attention 과정을 거치게 된다면, ***it*** 과 다른 단어와의 관계 정보가 포함된 벡터를 출력하게 되며, 여기서는 ***animal*** 단어와 굉장히 밀접한 관계를 가지고 있음을 내포하게 된다.

***'The animal didn't cross the street beacuase it was too tired.'***

<center>
<image src = https://user-images.githubusercontent.com/48677363/107319607-756a8380-6ae2-11eb-9dc6-b92e086cf6ee.png width = 300>
</center>

새로운 벡터를 출력하기 위해서 self-attention 구조는 기본적으로 **Query**, **Key**, **Value** 라는 3가지 벡터를 만들어내게 된다. 각 입력 데이터마다 **Q, K, V** 벡터가 생성되고 3개의 벡터의 계산을 통해 하나의 임베딩 벡터가 출력되게 된다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/107320373-e9595b80-6ae3-11eb-984f-e3473db4d262.png width = 500>
</center>

```python
n_batch, d_K, d_V = 3,128,256 # n_batch는 입력 데이터 N개
n_Q, n_K, n_V = 30,50,50
Q = torch.rand(n_batch,n_Q,d_K) # Q 벡터는 각 데이터 30 * 128 차원의 벡터로 표현
K = torch.rand(n_batch,n_K,d_K) # K 벡터는 각 데이터 50 * 128 차원의 벡터로 표현
V = torch.rand(n_batch,n_V,d_V) # V 벡터는 각 데이터 50 * 256 차원의 벡터로 표현
```

먼저, 입력 데이터의 Q, K, V 벡터를 통해 score 벡터를 생성해준다

  - score 벡터를 계산할 때, 인코딩을 하고자하는 단어의 Q 벡터와 나머지 단어의 K 벡터를 내적함으로써 관계도 혹은 유사도를 구할 수 있다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/107320832-d1cea280-6ae4-11eb-95cc-968cf250fe52.png width = 500>
</center>

  - Q 벡터와 나머지 모든 데이터의 K 벡터가 내적된 상태로 값이 개별로 존재하게 된다면, hyperparameter로 정해진 차원의 루트로 정규화(Normalize)해준다. 그 이후에 score 벡터를 softmax를 취함으로써 sum to 1이 되게 만들어주면서 다른 단어와의 interaction과 같은 스칼라를 표현할 수 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107321523-0b53dd80-6ae6-11eb-9e70-6ebb1fef7892.png width = 450>
</center>

  - 각각의 정규화, softmax를 거친 score 값을 각 V 벡터를 곱하고 더해서(weighted-sum) 하나의 스칼라로 나오는 것이 해당 단어의 attention value라고 할 수 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107322013-fb88c900-6ae6-11eb-9730-b29f527d7113.png width = 500>
</center>

  - 최종적으로 self-attetnion으로 벡터를 표현하고자하는 입력 데이터의 attention value를 다음과 같이 정리할 수 있다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107323859-612a8480-6aea-11eb-825c-a85f3b935a39.png width = 500>
</center>

**transformer가 보다 효과적으로 sequential data를 임베딩하여 표현할 수 있는 이유는 다음과 같다. 이미지를 MLP 혹은 CNN을 통해 벡터로 표현하고자 할 때, 신경망 구조가 동일하다면, 동일한 이미지에서는 동일한 출력이 나오게 된다. 하지만 transformer의 경우에는 같은 단어 혹은 토큰임에도 주변 단어와 데이터에 따라 출력값이 달라지기 때문에 보다 풍부한 표현이 가능하다고 볼 수 있다.**

#### 4) Multi-head attention

Multi-head attention은 말그대로 attention 과정을 여러번 거치는 것이다. 하나의 sequential data 셋에 Q, K, V 벡터를 한번만 생성하는 attention 과정을 한번만 거치는 것이 아닌 N번 수행함으로써 Q, K, V 벡터도 N번 생성되는 것을 의미한다.

Multi-head attention을 함으로써 서로 다른 N개의 임베딩된 벡터(결과)를 얻을 수 있게 되며 concat 후 weight matrix를 내적함으로써 최종 attention values를 구할 수 있게 된다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/107324959-4527e280-6aec-11eb-8e51-72e65c86fc94.png width = 600>
</center>

**Positional encoding**

positional encoding이 필요한 이유는 sequential data에 대한 정보가 attention 과정에 녹아들어야 하기 때문이다. 사실 attention 과정은 순서에 독립적이기 때문에 [a, b, c, d, e]와 [b, c, d, e, a]가 동일한 출력값을 가지게 된다.

pre-defined 된 벡터를 look-up 하는 형식으로 벡터를 더해줌으로써 positional한 정보를 포함하게 만들어준다.

<br>

**Decoder**


<image src = http://jalammar.github.io/images/t/transformer_decoding_1.gif width = 600>

<image src = http://jalammar.github.io/images/t/transformer_decoding_2.gif width = 600>

----------


### Further Question

LSTM에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있습니다. 무엇일까요?
Pytorch LSTM 클래스에서 3dim 데이터(batch_size, sequence length, num feature), `batch_first` 관련 argument는 중요한 역할을 합니다. `batch_first=True`인 경우는 어떻게 작동이 하게되는걸까요?