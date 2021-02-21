# Week04 - NLP, Natural Langurage Processing

## [Day 18] - Sequence to Sequence

### 1. Sequence to Sequence with Attention

이번 강의에서는 Seqeunce 데이터를 Encoder와 Decoder가 존재하는 Sequence to Sequence 구조에 대해서 배우게 됩니다.

Seq2Seq 구조는 encoder와 decoder로 이루어져 있는 framework로 대표적인 자연어 처리 구조 중 하나입니다. Encoder와 Decoder로는 다양한 알고리즘이 사용될 수 있지만 RNN과 Attention이 결합된 Seq2Seq 모델에 대해 알아보게 됩니다.

앞선 강의에서 다뤘듯이, RNN 모델이 가지고 있는 한계점을 보완하고자 Attention 매커니즘이 등장하게 되었습니다. 다양한 Attention 매커니즘의 종류와 이를 활용한 translation task에 대해서 알아볼 수 있습니다.

### 1) Seq2Seq Model 개념

Seq2Seq 모델은 기계번역에서 대표적으로 사용되는 모델로 input으로 토큰 및 단어의 sequence가 입력되면 output 또한 단어 혹은 토큰의 sequence가 출력되는 RNN의 many-to-many 모델에 속하게 됩니다.

Seq2Seq2 모델에서 Ecoder는 input 데이터 sequence에 대해 처리하고 Decoder는 output 데이터의 출력을 담당합니다. 두 개의 모듈은 parameter를 공유하지 않는 개별적인 RNN 모듈이라고 할 수 있는데 아래의 예시를 통해 확인할 수 있습니다.

아래의 그림은 chatbot과 같은 dialog system의 Seq2Seq 모델입니다. Encoder에 입력되는 'Are you free tomorrow?' 의 sequence data의 마지막 hidden state의 정보가 Decoder의 H_0으로 입력됩니다. 이 때, 두 모듈은 parameter를 공유하지 않는 RNN 모듈이며, Decoder는 H_0을 입력받게 되면서 다음 단어를 predict하게 되고 ground truth와 비교를 통해 loss를 줄여나갑니다.

ground truth에 해당하는 sequential data가 Decoder에 입력될 때에는, sequential data의 시작을 알리는 <start> 토큰과 끝을 알리는 <end> 토큰을 붙임으로써 예측이 반복됩니다.

<image src = https://user-images.githubusercontent.com/48677363/108620537-1efd2d80-7470-11eb-83dc-224d002e5d8a.png width = 600>

### 2) Seq2Seq Model with Attention

Attention 매커니즘은 RNN이 선천적으로 가지고 있는 long-term dependency problem과 bottlenect problem을 해결할 수 있습니다. Sequential data에서 모든 과정을 거친 마지막 hidden state vactor만 을 사용했던 LSTM과는 달리 Attention은 Decoder의 각 step에서 source sequence의 특정 부분에 대한 집중(attention) 정도를 파악함으로써 이 문제에 대한 해결책을 제안할 수 있었습니다.

<image src = https://user-images.githubusercontent.com/48677363/108621163-67b6e580-7474-11eb-8c36-3daf2449ad93.png width = 400>

  - Encoder에는 동일하게 각 sequence의 입력별로 $h^{e}_{t}$가 출력됨
  - Encoder의 마지막 $h^{e}_{t}$는 Decoder의 $h^{d}_{0}$로 입력됨
  - Encoder의 정보가 담긴 



------------

### 2. Beam Search and BLEU

이번 강의에서는 문장을 decoding 하는 과정에 대표적으로 사용하는 알고리즘인 Beam Search와 번역 task에서 대표적인 metric으로 사용되는 BLEU score에 대해 배우게 됩니다.

언어 모델이 문장을 generation할 때에는 확률값에 기반한 다양한 경우의 수가 존재합니다. 모든 경우의 수를 고려하는 것은 비효율적이며 너무 작은 확률값까지 고려한다면 생성된 문장의 질이 떨어질 수 있습니다. 가장 높은 확률값을 고려하는 방법 역시 모델이 단순한 generation에 불과해진다는 단점이 있습니다. 이러한 문제를 해결하기 위한 대안으로 Beam Search가 제안되었습니다.

자연어는 컴퓨터가 이해할 수 있는 벡터로 변환되어 모델의 입력 및 출력으로 활용되기 때문에 적합한 metric으로 모델을 평가해야합니다. 다양한 자연어 처리 관련 metric이 존재하지만, 그 중에서도 번역 task에서 가장 대표적인 BLEU score가 있으며 BLEU score가 어떻게 번역 task의 metric으로 활용되는 지에 대해 고민해보는 것이 좋습니다.



