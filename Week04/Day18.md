# Week04 - NLP, Natural Langurage Processing

## [Day 18] - 

### 1. Sequence to Sequence with Attention

이번 강의에서는 Seqeunce 데이터를 Encoder와 Decoder가 존재하는 Sequence to Sequence 구조에 대해서 배우게 됩니다.

Seq2Seq 구조는 encoder와 decoder로 이루어져 있는 framework로 대표적인 자연어 처리 구조 중 하나입니다. Encoder와 Decoder로는 다양한 알고리즘이 사용될 수 있지만 RNN과 Attention이 결합된 Seq2Seq 모델에 대해 알아보게 됩니다.

앞선 강의에서 다뤘듯이, RNN 모델이 가지고 있는 한계점을 보완하고자 Attention 매커니즘이 등장하게 되었습니다. 다양한 Attention 매커니즘의 종류와 이를 활용한 translation task에 대해서 알아볼 수 있습니다.






------------

### 2. Beam Search and BLEU

이번 강의에서는 문장을 decoding 하는 과정에 대표적으로 사용하는 알고리즘인 Beam Search와 번역 task에서 대표적인 metric으로 사용되는 BLEU score에 대해 배우게 됩니다.

언어 모델이 문장을 generation할 때에는 확률값에 기반한 다양한 경우의 수가 존재합니다. 모든 경우의 수를 고려하는 것은 비효율적이며 너무 작은 확률값까지 고려한다면 생성된 문장의 질이 떨어질 수 있습니다. 가장 높은 확률값을 고려하는 방법 역시 모델이 단순한 generation에 불과해진다는 단점이 있습니다. 이러한 문제를 해결하기 위한 대안으로 Beam Search가 제안되었습니다.

자연어는 컴퓨터가 이해할 수 있는 벡터로 변환되어 모델의 입력 및 출력으로 활용되기 때문에 적합한 metric으로 모델을 평가해야합니다. 다양한 자연어 처리 관련 metric이 존재하지만, 그 중에서도 번역 task에서 가장 대표적인 BLEU score가 있으며 BLEU score가 어떻게 번역 task의 metric으로 활용되는 지에 대해 고민해보는 것이 좋습니다.



