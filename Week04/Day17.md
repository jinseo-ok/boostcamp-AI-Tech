# Week04 - NLP, Natural Langurage Processing

## [Day 17] - RNN

### 1. Recurrent Neural Network and Language Modeling

이번 강의에서는 자연어 처리 분야에서 RNN을 활용하는 다양한 접근법과 이를 이용한 Language Model에 대해서 배우게 됩니다.

RNN 알고리즘은 입력의 순서를 고려하고 표현하기 위해 대체로 사용되어 왔습니다. RNN 구조를 활용하게 된다면 대표적인 sequential data인 NLP의 다양한 분야에서 문제를 정의하고 해결할 수 있습니다.

Language Model은 이전에 등장한 단어를 condition(조건)으로 다음에 등장할 단어를 예측하는 모델입니다. 이전 단어를 condition으로 학습하기 위해서 다양한 neural network 알고리즘을 활용할 수 있습니다. 이번 강의에서는 특히 RNN을 이용한 character-level의 language model에 대해서 배우게 됩니다.

앞 선 강의에서, RNN 또한 문제점이 있다고 했습니다. 초반 time step의 정보를 전달하기 어려운 점과 gradient vanishing / exploding 문제의 개념과 해결 방안 등에 대해서 한번 더 짚고 넘어가는 것이 좋습니다.

#### 1) RNN 구조

RNN의 기본적인 구조와 모듈 내 계산이 이뤄지는 파라미터는 다음과 같습니다. RNN은 sequential data가 순서대로 입력되며 이 전 시점의 정보를 $h_{t-1}$로 유지하며 네트워크가 진행됩니다.

모든 time step마다 입력되는 데이터는 동일한 RNN 네트워크를 거치게 되는데, 이는 **모두 동일한 weight를 공유**하고 있음을 의미합니다. 즉, 입력되는 모든 sequential data의 정보가 반영된 weight가 학습됨을 말합니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108050916-99533980-708d-11eb-8529-601265516375.png width = 300>
</center>

  - $h_{t-1}$: $t$ 시점 이전의 RNN 모듈에서 계산된 출력값
  - $x_{t}$: $t$ 시점에 입력되는 데이터
  - $h_{t}$: $h_{t-1}$과 $x_{t}$로 입력되어 RNN 모듈로 계산되는 출력값
  - $f_{W}$: RNN 모듈 
  - $y_{t}$: 모든 시점에서 출력되는 값

<center>
<image src = https://user-images.githubusercontent.com/48677363/108052063-f56a8d80-708e-11eb-90b9-f54341d04452.png width = 600>
</center>

t 시점의 데이터 $x_{t}$와 이 전 시점의 출력값인 $h_{t-1}$이 RNN 네트워크로 입력되면 linear transformation을 통해 $h_{t}$가 계산됩니다.

간단하게 $x_{t}$를 3차원 벡터로, $h_{t-1}$를 2차원으로 가정한다면 총 5차원인 입력 데이터가 2차원인 $h_{t}$로 계산되기 위해서는 총 2 * 5 차원의 weight matrix가 형성되야 합니다. 이 때, 2 * 3 차원의 weight는 $x_{t}$와 내적되는 $W_{xh}$이며, 2 * 2 차원의 weight는 $h_{t-1}$와 내적되는 $W_{hh}$ 입니다. 우측 아래 그림을 보게 되면 보다 이해가 쉬울 것입니다..

#### 2) RNN 종류

neural network는 입력과 출력의 차이로 다양한 구조를 가지게 됩니다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/108053146-62325780-7090-11eb-9701-e31338067860.png width = 500>
</center>

  - **one to one**: 가장 일반적인 네트워크 구조로, 입력에 따라 출력이 매핑되며 sequential data를 처리하기 힘듭니다.
  - **one to many**: image captioning 과제에 보통 사용되며, 하나의 이미지를 입력하게 됩니다. 진행되는 과정에 zero shape으로 이루어진 데이터를 입력할 때도 있습니다.
  - **many to one**: text와 같은 sequential data가 입력될 수 있는 구조이며, 마지막 네트워크에서 출력값이 형성됩니다. 감정 분석 등에 사용됩니다.
  - **many to many**: sequential data가 순서대로 입력되며 출력 또한 그에 맞게 출력되는 경우이며, 기계 번역 혹은 pos tagging과 같은 과제에서 사용됩니다.

#### 3) Character-level Language Model

Language Model은 입력되는 토큰 혹은 단어를 바탕으로 다음 단어를 예측하는 과제로, word-level은 단어를 예측하는, character-level은 글자를 예측하는 과제라고 할 수 있습니다.

Character-level Language Model은 sequential한 특징을 고려한 many-to-many RNN 구조로 이해할 수 있으며, 매 step 마다 바로 결과가 출력되게 됩니다.

이 때, output layer에서 출력되는 값을 softmax 취해줌으로써 각 클래스(단어)에 속할 확률로 학습이 이뤄지게 됩니다.

![image](https://user-images.githubusercontent.com/48677363/108054952-acb4d380-7092-11eb-96d0-984d442418a0.png)

**Backpropagation through time (BPTT)**

output layer에서 이뤄지는 softmax loss를 계산하며 weight를 업데이트하는 backpropagation 과정이 이뤄지게 됩니다. 그런데 sequential data의 길이가 길게 되면 업데이트해야할 파라미터가 매우 많아지기 때문에 모든 시점의 네트워크의 loss를 고려하여 업데이트를 진행하는 것이 아닌 truncated(삭제)하여 일부만을 선정하여 진행하게 됩니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108056252-5fd1fc80-7094-11eb-9384-561b9dbf7d10.png width = 500>
</center>

#### 4) RNN 문제점

RNN은 sequential data를 고려하여 결과를 출력하는 네트워크지만 그에 수반하는 문제점이 존재합니다. 네트워크가 연속적으로 영향을 받으며 계산되기 때문에 필연적으로 gradient가 증감 혹은 증폭하는 gradient vanshing / exploding 문제가 발생합니다.

아래 예시를 보면, weight가 3으로 유지될 때, 모든 step의 RNN 네트워크가 chain rule처럼 연쇄적으로 계산되기 때문에 3의 거듭제곱으로 기울기가 증폭하게 됩니다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/108056582-cbb46500-7094-11eb-8eee-5bbae8d24685.png width = 500>
</center>

----------

### 2. LSTM and GRU

이번 강의에서는 RNN이 본질적으로 가지고 있던 문제점을 개선하기 위해 등장했던 LSTM과 GRU에 대해 보다 구체적이게 알아보게 됩니다.

LSTM과 GRU가 gradient flow를 개선할 수 있는 이유에 대해 알아보게 된다면 이 두 알고리즘의 특징을 보다 이해할 수 있습니다.

RNN이 본질적으로 가지고 있던 문제점 중 하나는 바로 sequential data가 길어져 time step가 멀어질수록 정보를 잃어 학습능력이 저하되는 vanishing gradient problem가 있습니다.

이 문제를 극복하기 위해서 Long Short-Term Memory인 LSTM이 고안되었으며 cell state를 추가함으로써 정보를 보다 유지하는 구조를 보입니다.


#### 1) LSTM

<center>
<image src = https://user-images.githubusercontent.com/48677363/108058867-fb18a100-7097-11eb-9f2c-1c8b5c486f7c.png width = 350>
</center>

LSTM은 RNN과 달리 입력되는 데이터가 $x_{t}, c_{t-1}, h_{t-1}$로 이루어져 있습니다. 또한 내부적으로 총 4개의 linear transformation이 발생하므로 각기 다른 4개의 weigt를 가지고 있습니다.

LSTM은 $c_{t}$라는 cell state를 통해 이 전 RNN 네트워크에서 발생했던 정보들을 유지 및 탈락시킴으로써 보다 효과적으로 sequential data를 활용하고자 합니다.

먼저 cell state를 계산하기 위해 $h_{t-1}$과 $x_{t}$ 총 4번의 각기 다른 계산이 발생하게 됩니다. 아래 그림을 보게 되면 입력되는 두 데이터에 내적되는 weight의 각 차원과 역할에 따라 업데이트됩니다.


<image src = https://user-images.githubusercontent.com/48677363/108058613-9eb58180-7097-11eb-8826-d035bec9d273.png width = 500>


  - **Forget gate**

어떤 정보를 버릴지 선택하는 구간인 forget gate는 linear transformation의 결과인 logit과 element-wise product 되면서 logit의 양에 따라 $C_{t-1}$에서 넘어온 정보가 남게 됩니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108060091-c4438a80-7099-11eb-849c-239f94c01e15.png width = 330>
</center>

  - **Input gate & Update cell**

input gate는 각 weight에 따라 계산되는 것은 동일합니다. 여기서 LSTM의 핵심인 cell state가 update되는 과정이 발생합니다. forget gate에서 이뤄진 결과와 input gate에서 출력된 결과와 multiply되면서 다음 LSTM 네트워크로 이동할 t 시점의 $c_{t}$가 생성됩니다.

<image src = https://user-images.githubusercontent.com/48677363/108060716-9f034c00-709a-11eb-8e29-b50811986e58.png width = 450>

#### 2) GRU

GRU는 LSTM이 추구하고자 하는 방향과 굉장히 동일합니다. 하지만 cell state와 hidden state를 일원화함으로써 계산되는 과정을 축소시켰습니다. 특히 LSTM에서는 cell state가 업데이트되는 과정 중, forget gate의 결과와 input gate의 결과가 독립적인 상태에서 multiply되었지만 **GRU는 $z_{t}$와 $1 - z_{t}$를 곱해줌으로써 정보의 가중평균을 활용**했다고 볼 수 있습니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108061294-6f087880-709b-11eb-893c-05592e5aab43.png width = 300>
</center>