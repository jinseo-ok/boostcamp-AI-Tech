# Week04 - NLP, Natural Langurage Processing

## [Day 16] - Introduction to NLP

### 1. Intro to NLP, Bag-of-Words

NLP, 자연어처리의 소개를 시작으로 가장 간단한 모델 중 하나인 Bag-of-Words의 개념에 대해서 다루게 된다.

Bag-of-Words는 one-hot-encoding을 사용하여 단어를 표현하고 단어의 순서를 고려하지 않는 굉장히 간단한 방법이다. 매우 간단한 모델이지만 많은 자연어처리 과제에서 효과적으로 동작하는 알고리즘 중 하나이다. 그리고 Bag-of-Words를 이용해 문서 분류 과제에 활용하는 Naive Bayes Classifier에 대해서도 다루게 된다. 

이번 강의에서는 단어가 벡터로 표현되는 방법, 문서가 벡터로 표현되는 방법에 대해 유념하며 듣는 것이 중요하다.

#### 1) NLP 분야

자연어처리는 기본적으로 컴퓨터가 주어진 단어나 문장 혹은 문단과 글을 이해하는 Natural Language Understanding(NLU)와 자연어를 상황에 따라 적절히 생성하는 Natural Language Generation(NLG)로 구분된다.

또한, 자연어처리 분야는 컴퓨터비전 분야와 더불어 인공지능 및 딥러닝 기술이 가장 활발하게 연구되고 발전하고 있다. 

자연어처리 분야에서도 학문적인 체계로 보면 자연어를 다루는 다양한 분야가 존재한다.

**Natural language processing**

  - Low-level parsing
    - Tokenization: 문장을 이해하기 위해, 문장을 이루는 단어를 정보로 인식하여 단어 단위 혹은 의미 단위로 쪼개는 과정
    - stemming: 형태소 분석 중 하나인 어간 추출을 의미하며, 어미를 자름으로써 동일한 어간을 추출하는 과정
<br>
  - Word and phrase level
    - Named entity recognition(NER): 개체명 인식으로 고유명사를 인식하기 위한 과정
    - part-of-speech(POS): 문장 내 단어들의 품사 혹은 성분을 인식하는 과정
<br>
  - Sentence level
    - Sentiment analysis: 주어진 문장을 긍정 혹은 부정 어조로 분류하는 과정
    - machine translation: 기계번역으로 주어진 문장을 다른 언어의 문장으로 번역하는 과정
<br>
  - Multi-sentence and paragraph level
    - Entailment prediction: 두 문장간의 논리적인 내포 혹은 모순 관계를 예측하는 과정
    - question answering: 독해기반 질의응답으로 질문을 정확하게 이해하고 가지고 있는 데이터를 기반으로 답을 제시하는 과정
    - dialog systems: chat-bot과 같은 대화를 수행할 수 있는 과정
    - summarization: 주어진 문서를 요약하는 과정

**Text Mining**

  - 방대한 규모의 텍스트 데이터에서 유용한 정보와 insight를 추출하는 과정
  - topic modeling을 통한 문서 군집화
  - 텍스트 데이터 분석을 통한 사회현상 혹흔 사회문화와의 논리적인 연관성 분석

**Information retrieval**

  - 정보 검색 기술에 대한 연구가 진행되는 분야
  - 정보 검색 분야에 포함되는 추천 시스템 분야가 최근에는 빠르게 관심을 받고 있으며 발전하고 있음
  - 추천 시스템은 다양한 분야에서 활용되고 있으며 비즈니스적인 부분에서 굉장히 각광받고 있음

#### 2) NLP 동향

**Word Embedding**

먼저 **자연어**란 '사람들이 일상적으로 쓰는 언어를 인공적으로 만들어진 언어인 인공어와 구분하여 부르는 개념'이다. 그렇기 때문에 컴퓨터는 자연어 그 자체를 인식하여 이해할 수 없기 때문에 컴퓨터가 이해할 수 있는 표현법으로 자연어를 변환해야 한다.

자연어를 특정한 차원으로 이루어진 벡터로 표현하는 과정을 거치게 되며, 단어 혹은 토큰을 벡터 공간의 한 점으로 표현한다는 의미로 **Word Embedding, 워드 임베딩** 이라고 한다.

**RNN**

자연어는 소통을 함에 있어 시작과 끝이라는 순서가 존재하는 데이터로 같은 단어 혹은 토큰임에도 입력되는 순서에 따라 가지는 의미가 달라질 수 있다. 이 말은 즉, 같은 단어와 토큰이 워드 임베딩 과정을 거치게 되었을 때, sequence에 따라 워드 임베딩 벡터가 달라질 수 있음을 의미한다. 그렇기 때문에 sequence를 고려한 RNN 모델이 자연어 처리의 핵심 모델로 통상적으로 사용되었다.

**Transformer**

2017년 구글에서 'Attention is all you need' 논문이 발표되면서 기존의 RNN 기반의 자연어 처리 모델 구조를 self-attention 구조의 **Transformer**라는 새로운 모델이 등장하게 되었다. 다양한 자연어 처리 분야에서 Transformer 모델의 성능이 확인되면서 현재 대부분의 자연어 처리를 위한 모델은 Transformer 모델을 기반으로 하고 있다.

Transformer 모델의 시작은 기계 번역에 초점을 맞추고 있었지만 현재에는 자연어 처리의 다양한 분야를 넘어, 영상 처리, 시계열 예측, 신약 개발, 신물질 개발 등 매우 다양한 분야에서 활용되고 있으며 성능 향상을 이뤄내고 있다.

#### 3) Bag-of-Wrods





------

### 2. Word Embedding


--------
