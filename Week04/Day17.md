# Week04 - NLP, Natural Langurage Processing

## [Day 16] - RNN

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



----------

### 2. LSTM and GRU

이번 강의에서는 RNN이 본질적으로 가지고 있던 문제점을 개선하기 위해 등장했던 LSTM과 GRU에 대해 보다 구체적이게 알아보게 됩니다.

LSTM과 GRU가 gradient flow를 개선할 수 있는 이유에 대해 알아보게 된다면 이 두 알고리즘의 특징을 보다 이해할 수 있습니다.


<image src = https://user-images.githubusercontent.com/48677363/108050165-97d54180-708c-11eb-85c2-af0eb642332c.png width = 400>
