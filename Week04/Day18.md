# Week04 - NLP, Natural Langurage Processing

## [Day 18] - Sequence to Sequence

### 1. Sequence to Sequence with Attention

이번 강의에서는 Seqeunce 데이터를 Encoder와 Decoder가 존재하는 Sequence to Sequence 구조에 대해서 배우게 됩니다.

Seq2Seq 구조는 encoder와 decoder로 이루어져 있는 framework로 대표적인 자연어 처리 구조 중 하나입니다. Encoder와 Decoder로는 다양한 알고리즘이 사용될 수 있지만 RNN과 Attention이 결합된 Seq2Seq 모델에 대해 알아보게 됩니다.

앞선 강의에서 다뤘듯이, RNN 모델이 가지고 있는 한계점을 보완하고자 Attention 매커니즘이 등장하게 되었습니다. 다양한 Attention 매커니즘의 종류와 이를 활용한 translation task에 대해서 알아볼 수 있습니다.

#### 1) Seq2Seq Model 개념

Seq2Seq 모델은 기계번역에서 대표적으로 사용되는 모델로 input으로 토큰 및 단어의 sequence가 입력되면 output 또한 단어 혹은 토큰의 sequence가 출력되는 RNN의 many-to-many 모델에 속하게 됩니다.

Seq2Seq2 모델에서 Ecoder는 input 데이터 sequence에 대해 처리하고 Decoder는 output 데이터의 출력을 담당합니다. 두 개의 모듈은 parameter를 공유하지 않는 개별적인 RNN 모듈이라고 할 수 있는데 아래의 예시를 통해 확인할 수 있습니다.

아래의 그림은 chatbot과 같은 dialog system의 Seq2Seq 모델입니다. Encoder에 입력되는 'Are you free tomorrow?' 의 sequence data의 마지막 hidden state의 정보가 Decoder의 H_0으로 입력됩니다. 이 때, 두 모듈은 parameter를 공유하지 않는 RNN 모듈이며, Decoder는 H_0을 입력받게 되면서 다음 단어를 predict하게 되고 ground truth와 비교를 통해 loss를 줄여나갑니다.

ground truth에 해당하는 sequential data가 Decoder에 입력될 때에는, sequential data의 시작을 알리는 <start> 토큰과 끝을 알리는 <end> 토큰을 붙임으로써 예측이 반복됩니다.

<image src = https://user-images.githubusercontent.com/48677363/108620537-1efd2d80-7470-11eb-83dc-224d002e5d8a.png width = 600>

#### 2) Seq2Seq Model with Attention

Attention 매커니즘은 RNN이 선천적으로 가지고 있는 long-term dependency problem과 bottlenect problem을 해결할 수 있습니다. Sequential data에서 모든 과정을 거친 마지막 hidden state vactor만 을 사용했던 LSTM과는 달리 Attention은 Decoder의 각 step에서 source sequence의 특정 부분에 대한 집중(attention) 정도를 파악함으로써 이 문제에 대한 해결책을 제안할 수 있었습니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108621556-b796ac00-7476-11eb-9a87-becd746cc024.png width = 500>
</center>

  - Encoder에는 동일하게 각 sequence의 입력별로 $h^{e}_{t}$가 출력됨
  - Encoder의 마지막 $h^{e}_{t}$는 Decoder의 $h^{d}_{0}$로 입력됨
  - Encoder의 정보가 담긴 $h^{d}_{0}$과 $x^{d}_{1}$를 입력으로 $h^{d}_{1}$을 출력함
  - $h^{d}_{1}$는 다음 $x^{d}_{2}$를 예측하는 데에 사용할 뿐만 아니라 Encoder의 $h^{e}_{t}$와의 attention scores를 모두 구하게 됨
  - 이 때, $h^{d}_{1}$와 Encoder의 $h^{e}_{t}$의 attention score를 구하는 과정은 유사도를 구하는 개념과 유사하면 dot-product가 사용될 수 있음
  - 계산된 유사도 모두 softmax를 통과함으로써 가중치로 사용할 수 있는 확률값을 구할 수 있으며, 이를 기반으로 가중 평균하여 output으로 **Attention value*를 얻을 수 있음
  - 계산된 Attention value와 $h^{d}_{t}$를 concat한 뒤에 다음 step의 단어를 예측하는 데에 사용함
  - 즉, Attention value는 모든 $h^{d}_{t}$가 가지는 각기 다른 값이 되며, Encoder의 data와의 attention 정보가 담겼다고 할 수 있음

**Teacher forcing**

  - Teacher forcing is a method for quickly and efficiently training recurrent neural network models that use the ground truth from a prior time step as input.
  - Teacher Forcing is the technique where the target word is passed as the next input to the decoder

Teacher forcing은 주로 Encoder와 Decoder 구조의 Seq2seq 모델에서 주로 사용되는 기법입니다. Decoder의 매 step으로 입력하는 sequential data를 ground truth로 대체하는 기법을 **Teacher forcing** 이라고 합니다. 이러한 사실을 기반으로 했을 때, Decoder의 매 step으로 입력되는 2가지 방식이 있음을 알 수 있습니다.

  - 매 step의 입력 데이터로 기존의 ground truth를 입력하는 방식
  - 매 step의 입력 데이터로 이 전 step의 prediction을 입력하는 방식

<image src = https://user-images.githubusercontent.com/48677363/108622392-d8adcb80-747b-11eb-9b2f-aec4f89ecc4a.png>

Teacher forcing을 사용하게 되면 ground truth를 사용해 예측이 가능하기 때문에 초기에 학습이 빠르게 가능하다는 장점이 있습니다. 하지만 추론 과정에서는 ground truth를 치팅할 수 없기 때문에 오히려 모델의 성능이 보장될 수는 없다는 단점이 있습니다.

그렇기 때문에 강의에서는 학습 초기에는 teacher forcing으로 학습을 진행한 후에 후반부에는 모델의 예측값을 입력으로 사용하는 방법을 적용할 수 있습니다.

**Attention socores**

앞 서 Decoder의 입력 데이터에 대한 hidden state, $h^{d}_{t}$와 Encoder의 모든 $h^{e}_{t}$ 집합과의 attention score를 계산하여 유사도를 구함으로써 해당 데이터가 어디에 보다 attention 되었는지 확인 가능하다 했습니다.

이 때, 유사도를 의미하는 attention score를 구하는 방법으로 내적하는 방법을 소개했지만 그 외에도 다양한 방법을 적용할 수 있습니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108622563-f62f6500-747c-11eb-9256-d1a53f590c55.png width = 400>
</center>

#### 3) Attention 매커니즘의 장점

- Attention significantly improves NMT performance
  - It is useful to allow the decoder to focus on particular parts of the source

- Attention solves the bottleneck problem
  - Attention allows the decoder to look directly at source; bypass the bottleneck

- Attention helps with vanishing gradient problem
  - Provides a shortcut to far-away states

- Attention provides some interpretability
  - By inspecting attention distribution, we can see what the decoder was focusing on
  - The network just learned alignment by itself

------------

### 2. Beam Search and BLEU

이번 강의에서는 문장을 decoding 하는 과정에 대표적으로 사용하는 알고리즘인 Beam Search와 번역 task에서 대표적인 metric으로 사용되는 BLEU score에 대해 배우게 됩니다.

언어 모델이 문장을 generation할 때에는 확률값에 기반한 다양한 경우의 수가 존재합니다. 모든 경우의 수를 고려하는 것은 비효율적이며 너무 작은 확률값까지 고려한다면 생성된 문장의 질이 떨어질 수 있습니다. 가장 높은 확률값을 고려하는 방법 역시 모델이 단순한 generation에 불과해진다는 단점이 있습니다. 이러한 문제를 해결하기 위한 대안으로 Beam Search가 제안되었습니다.

자연어는 컴퓨터가 이해할 수 있는 벡터로 변환되어 모델의 입력 및 출력으로 활용되기 때문에 적합한 metric으로 모델을 평가해야합니다. 다양한 자연어 처리 관련 metric이 존재하지만, 그 중에서도 번역 task에서 가장 대표적인 BLEU score가 있으며 BLEU score가 어떻게 번역 task의 metric으로 활용되는 지에 대해 고민해보는 것이 좋습니다.

#### 1) Beam Search

Decoding 과정에서 다음 step의 단어 및 토큰을 예측함에 있어 다양한 알고리즘이 존재합니다.

**Greedy Search**

Greedy Search는 해당 step에서 가장 확률이 높은 class를 선택하는 방법으로 이 전 혹은 이 후 step과 독립적입니다. Greedy Search의 장점으로는 argmax한 class만을 출력하면 되기때문에 매우 빠르지만 output 또한 sequence 개념으로 출력되야 하는 Seq2Seq 모델에서 각 step의 최대 확률이 전체 sequence의 최대 확률을 보장하지는 않습니다.

**Exhaustive Search**

완전 탐색이라고도 불리는 Exhaustive Search는 가능한 모든 경우를 고려하는 알고리즘입니다. 

<image src = https://user-images.githubusercontent.com/48677363/108623203-12350580-7481-11eb-81b1-f1e2294c25a5.png width = 500>

모든 step의 확률을 이전 step의 조건부확률을 고려함으로써 vocabulary size가 V일 때(class 개수가 V), 시간 복잡도는 $O(V^{t})$로 형성됩니다. Exhaustive Search의 경우 모든 경우를 탐색하면서 최선의 결과를 도출할 수 있는 가능성은 높아지지만 너무 오래 걸린다는 단점으로 오히려 비효율적일 수 있습니다.

**Beam Search**

앞서 살펴본 Greedy Search와 Exhaustive Search의 단점을 보완한 방법인 Beam Search는 효과적인 성능으로 RNN 학습과 자연어처리 분야에서 활발히 사용된다고 합니다.

Beam Search는 주어진 k를 기준으로 기억하는 노드의 수를 제한함으로써 효율성을 높인 방식입니다.

<image src = https://user-images.githubusercontent.com/48677363/110068715-9e1c2b00-7db9-11eb-98a3-1cd32f678418.png width = 1000>

(1) decoding 과정에서 $<start>$ 토큰이 주어지면 모델은 next word에 대한 예측을 하게 되고 k 기준에 따라 top next word를 선별합니다. 여기서 K는 2로 가정하게됩니다.
(2) 결정된 k개의 word에서 다시 k개의 next word를 각각 구하게 되면 k^2의 next word에 대한 확률을 얻을 수 있습니다.
(3) k^2의 케이스에서 높은 확률의 next word를 k개만 다시 선택함으로써 k개의 노드를 유지하게 됩니다.
(4) 2-3의 과정을 반복하면서 $<END>$ 토큰이 나올 때까지 진행합니다.
(5) Beam Search의 k개의 각 결과의 score를 구함으로써, 가장 높은 score를 최종 예측값으로 출력합니다. 이 때, sentence의 길이가 길게되면 score가 높을 수 있으므로 sentence 길이로 정규화해줍니다.

$$score(y_1,...,y_t) = \frac {1}{t} \sum^t_{i=1} logP_{LM}(y_i|y_1,...,y_{i_1}, x)$$

#### 2) BLEU score

자연어 생성 모델의 정확도를 평가하기 위해서는 다양한 평가 방법이 존재합니다. 보통 정확도를 평가할 때 사용되는 방법으로는 precision, recall, f-measure이 있습니다.

$Reference: \text {Half of my heart is in Havana ooh na na}$
- $Predicted_1 : \text {Half as my heart is in Obama ooh na}$
- $Predicted_2 : \text {Havana na in heart my is Half ooh of na}$

**Precision**

$$Precision = \frac {\text {correct words}}{\text {length of prediction}}$$

**Recall**

$$Recall = \frac {\text {correct words}}{\text {length of reference}}$$

**F - measure**

$$F - measure = \frac {precision \times recall}{\frac {1}{2}(precision + recall)}$$

<br>

|Metric|Model_1|Model_2|
|:----:|:-----:|:-----:|
|Precision|78%|100%|
|Recall|70%|100%|
|F-measure|73.78%|100%|


하지만 위와 같은 평가 방법은 단순 정확도만을 평가하기 때문에, 자연어의 sequential한 특징을 평가 방법에 반영하지 못하는 단점이 있습니다. 해당 예시만 보더라도 예측 문장의 순서가 모두 틀렸음에도 정확도만을 기준으로 평가했기 때문에 상이한 평가 결과가 도출됩니다.

**BiLingual Evaluation Understudy (BLEU)**

자연어의 sequential한 특징을 고려한 BLEU score는 예측값의 N-gram을 실제값과 비교함으로써 문장의 전체 구성요소를 평가하기 위한 방법입니다. 주어진 N을 기준으로 1~N 까지의 모든 gram size에 대한 precision을 고려하고 너무 짧은 예측값에 대한 패널티(가중치)를 부여하는 특징을 가지고 있습니다. 여기서 gram size란 연속되는 단어의 개수를 의미합니다.

n이 4로 주어졌을 때, BLEU score가 구해지는 과정에 대해서 알아보겠습니다.

$Reference: \text {Half of my heart is in Havana ooh na na}$
- $Predicted_1 : \text {Half as my heart is in Obama ooh na}$
- $Predicted_2 : \text {Havana na in heart my is Half ooh of na}$

(1) predicted_1의 경우에는, 9개 중 7개의 단어가 일치하므오 7/9, predicted_2는 10개의 단어가 모두 일치하므로 10/10
(2) predicted_1은 (my heart), (heart is), (is in), (ooh na)가 일치하며 predicted_2는 일치하는 단어 두쌍이 존재하지 않습니다.
(3) predicted_1은 (my heart is), (heart is in)가 일치합니다.
(4) 해당 gram_size를 4까지 반복하여 곱할 수 있습니다.
(5) 각 예측값에 대한 패널티를 구하게 되면, predicted_1은 min(1, 9/10), predicted_2는 min(1, 10/10)이 됩니다. 

$$BLEU = min(1, \frac {\text {length of prediction}}{\text {length of reference}})(\prod^N_{i=1}Precision_i)^{\frac {1}{N}}$$

<br>

|Metric|Model_1|Model_2|
|:----:|:-----:|:-----:|
|Precision (1-gram)|$\frac {7}{9}$|$\frac {10}{10}$|
|Precision (2-gram)|$\frac {4}{8}$|$\frac {0}{9}$|
|Precision (3-gram)|$\frac {2}{7}$|$\frac {0}{8}$|
|Precision (4-gram)|$\frac {1}{6}$|$\frac {0}{7}$|
|Brevity penalty|$\frac {9}{10}$|$\frac {10}{10}$|
|BLEU|52%|0%|

