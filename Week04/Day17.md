# Week04 - NLP, Natural Langurage Processing

## [Day 16] - RNN

### 1. Recurrent Neural Network and Language Modeling

이번 강의에서는 자연어 처리 분야에서 RNN을 활용하는 다양한 접근법과 이를 이용한 Language Model에 대해서 배우게 됩니다.

RNN 알고리즘은 입력의 순서를 고려하고 표현하기 위해 대체로 사용되어 왔습니다. RNN 구조를 활용하게 된다면 대표적인 sequential data인 NLP의 다양한 분야에서 문제를 정의하고 해결할 수 있습니다.

Language Model은 이전에 등장한 단어를 condition(조건)으로 다음에 등장할 단어를 예측하는 모델입니다. 이전 단어를 condition으로 학습하기 위해서 다양한 neural network 알고리즘을 활용할 수 있습니다. 이번 강의에서는 특히 RNN을 이용한 character-level의 language model에 대해서 배우게 됩니다.

앞 선 강의에서, RNN 또한 문제점이 있다고 했습니다. 초반 time step의 정보를 전달하기 어려운 점과 gradient vanishing / exploding 문제의 개념과 해결 방안 등에 대해서 한번 더 짚고 넘어가는 것이 좋습니다.






----------

### 2. LSTM and GRU

이번 강의에서는 RNN이 본질적으로 가지고 있던 문제점을 개선하기 위해 등장했던 LSTM과 GRU에 대해 보다 구체적이게 알아보게 됩니다.

LSTM과 GRU가 gradient flow를 개선할 수 있는 이유에 대해 알아보게 된다면 이 두 알고리즘의 특징을 보다 이해할 수 있습니다.