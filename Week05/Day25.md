# Week05 - Graph

## [Day 25] - GNN, Graph Neural Network

### 1. GNN, 그래프 신경망 I

이번 강의에서는 정점 표현 학습, Node Representation Learning 의 방법 중 한 가지인 **그래프 신경망, Graph Neural Network(GNN)** 에 대해서 배웁니다.

#### 1) 정점 표현 학습

정점 표현 학습이란 **그래프의 정점들을 벡터의 형태로 표현**하는 것이며, **정점 임베딩, Node Embedding** 이라고도 부릅니다.

정점 임베딩 방법으로는 크게 **변환식, Transductive** 방식과 **귀납식, Inductive** 방식으로 구분할 수 있습니다.

- 변환식 방법: 학습의 결과로 정점의 임베딩 자체를 출력값으로 얻는 방식
- 귀납식 방법: 학습의 결과로 정점을 임베딩 벡터로 변환하는 함수를 출력값으로 얻는 방식

Day 24에서 중점적으로 알아본 변환식 임베딩 방법은 다음과 같은 **한계점**을 갖습니다.

- 학습이 진행된 이후에 추가된 정점에 대해서는 임베딩을 얻을 수 없음
- 모든 정점에 대한 임베딩을 미리 계산하여 저장해야함(저장해두지 않으면 다시 불러올 수 없기 때문으로 이해했음)
- 정점이 속성(Attribute) 정보를 가진 경우에 이를 활용할 수 없음

**귀납식 임베딩 방법**

반면 출력으로 인코더, 즉 정점을 임베딩 벡터로 변환하는 함수를 출력값으로 얻는 **귀납식 임베딩 방법** 은 다음과 같은 장점을 갖습니다.

- 학습이 진행된 이후에 추가된 정점에 대해서도 임베딩을 얻을 수 있음
- 모든 정점에 대한 임베딩을 미리 계산하여 저장해둘 필요가 없음
- 정점이 속성 정보를 가진 경우에도 이를 활용할 수 있음, 속성 정보를 활용할 수 있도록 함수를 설계할 수 있음

#### 2) GNN 기초

**GNN 기본 구조**

GNN은 그래프와 정점의 속성 정보를 입력으로 받습니다. 그래프는 **인접 행렬 $V * V$의 A**이며, 각 정점 𝑢의 **속성(Attribute) 벡터는 $m$ 차원 벡터 $X_u$** 입니다. 이 때, 정점의 속성의 예시로는 온라인 소셜 네트워크에서 사용자의 **지역, 성별, 연령, 프로필 사진 등**과 논문 인용 그래프에서 사용된 **키워드에 대한 one-hot 벡터** 등이 있습니다.

GNN은 이웃 정점들의 정보를 집계하는 과정을 반복하여 임베딩을 얻습니다. 아래 그림에서는 대상 정점의 임베딩을 얻기 위해 이웃 정점과 이웃의 이웃 정점의 정보를 우측과 같이 도식화할 수 있습니다. 여기서 주의할 점은 2단계 이웃 정점의 정보를 집계하는 과정에서 단계를 무시하고 독립적으로 이웃의 정보를 중복으로 집계했음을 알 수 있습니다.

각 이웃의 정보를 집계하는 단계를 **층, Layer** 라고 하며, 각 층마다 임베딩을 얻습니다. 즉, 이전 층의 이웃 정보를 **집계**하여 임베딩을 얻게 되는데, 0th-layer인 입력층의 임베딩으로는 **정점의 속성 벡터**를 사용합니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/109525304-bad40c80-7af4-11eb-9fba-f9493ae4180d.png width = 500>
</center>

결론적으로 임베딩 벡터를 얻고자 하는 대상 정점 마다 이웃의 상태가 다르기 때문에 집계되는 정보가 상이합니다. 이렇게 대상 정점 별 집계되는 구조를 계산 그래프, Compuation Graph라고 합니다.

**집계 함수**

하지만 서로 다른 대상 정점 간에도 층 별 집계 함수는 동일합니다. 여기서 주의해야할 점은 해당 정점이 가지고 있는 **이웃의 크기가 다르기 때문에 각 층별로 공유하는 함수는 가변적인 입력에 대한 처리가 필요**합니다. 

<center>
<image src = https://user-images.githubusercontent.com/48677363/109525796-3afa7200-7af5-11eb-8fb7-8268ea26187f.png width = 500>
</center>

그렇기 때문에 각 층별로 적용되는 집계 함수는 **이웃 정점의 정보의 평균을 계산**하고 **신경망에 적용**하는 단계를 거치게 됩니다. 이 때, 정보의 평균을 계산하는 단계는 가변적인 입력의 크기를 동일하게 만들어주기 위한 과정입니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/109527299-d5a78080-7af6-11eb-835a-a96eae7619e4.png width = 500>
</center>

<br>
GNN의 기본적인 구조를 수식적으로 정리하면 다음과 같습니다. input layer에 입력되는 벡터는 이웃 정점의 속성 정보들로 hidden layer와 같은 집계 함수 layer로 입력되게 됩니다. 집계 함수가 존재하는 K개의 layer에서 반복적으로 계산이 진행되며 output layer에서 특정 정점의 임베딩 벡터가 출력됩니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/109529182-ce817200-7af8-11eb-85ae-0921e9438667.png width = 700>
</center>

**손실 함수**

GNN의 손실함수의 목표는 **정점 간 거리를 보존하는 것** 입니다. 그래프에서 정점 간 유사도를 계산하는 방법을 정의함에 따라 손실함수 또한 다양한 접근법으로 정의될 수 있습니다. 

아래의 손실함수는 정점 간 유사도를 구하기 위한 인접성 기반 접근법에 대한 손실함수로 Day 24에서 배웠던 손실함수 수식과 동일합니다.

<image src = https://user-images.githubusercontent.com/48677363/109529769-61221100-7af9-11eb-8bab-61594cb8b978.png width = 500>

한편, 후속 과제(Downstream Task)의 손실함수를 이용한 종단종(End-to-End) 학습도 가능합니다.(개인적으로는 임베딩 벡터에 대한 손실함수가 아닌 추후 다른 과제를 위해 정의된 손실함수로 학습이 진행될 수 있음으로 이해했습니다.)

예를 들어, 정점 분류, Node Classification이 최종 과제로 주어졌을 때에는, 임베딩 벡터가 정점 간 유사도를 보존하는 정도가 중요한 것이 아닌 **정점을 정확히 분류하는 정도**가 최종 목표가 될 것입니다.

이 경우에는 분류기의 손실함수인 교차 엔트로피(Cross Entropy)와 같은 손실함수를 전체 학습 과정의 손실함수로 사용할 수 있습니다.

GNN을 End-to-End 학습을 통한 분류는 변환적 정점 임베딩 이후 분류기를 학습하는 것보다 정확도가 대체로 높은 경향을 보였습니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/109531303-11444980-7afb-11eb-828b-73eeb77830eb.png width = 550>
</center>

**귀납식 방식인 GNN의 활용**

다시 정리하자면, GNN은 손실함수를 기준으로 loss를 줄여나감으로써 집계가 발생하는 각 layer의 weight를 업데이트함으로써 단순 정점 임베딩 벡터가 아닌 주어진 그래프를 통해 generalization된 임베딩 모델?을 얻을 수 있게 됩니다.

이 때, 학습에 사용할 대상 정점은 모든 정점이 아닌 일부 정점만을 사용할 수 있으며, 선택된 정점들의 계산 그래프를 통해 weight를 업데이트할 수 있습니다. 

- 학습을 통해 만들어진 weight와 설계된 구조로 학습에 사용되지 않은 정점의 임베딩 계산이 가능함

<image src = https://user-images.githubusercontent.com/48677363/109532344-5ddc5480-7afc-11eb-97c4-1287834c4b97.png width = 500>

- 학습 이후에 추가된 정점의 임베딩 계산이 가능함

- 학습된 GNN을 새로운 그래프에 적용 가능하기도함

#### 3) GNN의 변형

앞서 이웃 정점의 정보를 집계하기 위한 집계 함수로 평균을 언급했지만, 다양한 형태의 집계 함수로 변형이 가능합니다. 그리고 접근법의 방식에 따라 정보를 집계하는 특징이 다를 것 입니다.

**그래프 합성곱 신경망, Graph Convolutional Network, GCN**

GCN의 집계 함수를 GNN의 집계 함수와 비교하게 되면 크게 2부분에서 차이가 발생합니다. 수식적으로 형태 자체가 크게 변하지는 않았지만 정보를 집계하는 작은 차이가 큰 성능의 향상으로 이뤄지기도 했습니다.

- 학습 파라미터인 weight 부분입니다. GNN에서는 집계가 이뤄지는 부분과 이전 층의 임베딩에서 각각 parameter가 존재했지만 GCN에서는 하나의 paramter로만 학습이 진행됩니다. 

- 정규화 방법의 변화입니다. GNN에서는 정점 𝑣의 연결성만 고려했다면 GCN에서는 정점 𝑢, 𝑣 연결성의 기하평균을 사용했습니다. 

![image](https://user-images.githubusercontent.com/48677363/109534517-c1678180-7afe-11eb-8b99-da503e5cabd2.png)

**GraphSAGE**

GraphSAGE의 집계 함수는 AGG 함수와 concat에서 차이가 발생합니다.

- 집계 함수로 통과한 벡터와 이전 층의 벡터와 concat하게 된 후에 activation function을 통과하게 됩니다.

- AGG라는 새로운 기능이 추가되었는데 이전 층의 이웃 벡터를 입력함으로써 집계가 이뤄지는 부분입니다. 이 때 다양한 집계 함수가 사용될 수 있습니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/109535511-017b3400-7b00-11eb-86cc-9e9d3b410b0d.png width = 500>

<image src = https://user-images.githubusercontent.com/48677363/109535696-34252c80-7b00-11eb-9e22-baf2c18b24d0.png width = 350>
</center>

#### 4) CNN과의 비교

CNN 또한 GNN과 유사하게 이웃의 정보를 집계하는 과정을 반복합니다. 대표적으로 CNN은 이미지 처리를 위해 주로 사용되는데, 이웃 픽셀의 정보를 집계하여 이미지의 특징을 파악하는 과정을 반복합니다.

- CNN에서는 입력되는 이웃의 수가 균일하지만, GNN에서는 입력되는 이웃의 수가 가변적임

- CNN의 2차원 픽셀 행렬과 GNN의 2차원 인접 행렬의 의미가 굉장히 다름

    - CNN에서 주로 사용되는 인접 픽셀은 위치적인 인접을 의미하므로 유용한 정보를 담고 있을 가능성이 높음
    - 하지만 그래프의 인접 행렬은 행과 열의 순서가 임의로 결정되는 경우가 많으며 인접 원소 자체가 가지는 정보가 굉장히 제한적임(픽셀은 픽셀 정보를 의미하지만 )

-------

### 2. GNN, 그래프 신경망 II

이번 강의에서는 GNN의 심화 내용에 대해서 배웁니다.

GNN과 Attention, 그래프 임베딩의 그래프 풀링, 그리고 학습 시 발생하는 지나친 획일화(Over-Smoothing) 문제에 대해 다룹니다.

#### 1) GNN과 Attention, GAT, Graph Attention Network

기본적인 GNN은 **이웃들의 정보를 동일한 가중치로 평균**을 내며, GCN에서도 역시 단순히 **연결성을 고려한 가중치로 평균**을 출력합니다.

하지만 특정 정점의 이웃의 정보를 학습하는 과정에서 동일한 가중치 혹은 단순 연결성만을 고려한 가중치로는 보다 풍부한 벡터를 얻는 것에 한계가 있다고 판단하여 **그래프 어텐션 신경망, Graph Attention Network, GAT**가 제안되었습니다.

GAT는 실제 그래프에서는 이웃 별로 미치는 영향이 다를 수 있기 때문에 가중치 자체도 학습이 이뤄집니다. 이 때, 가중치를 학습하기 위해서 Self-Attention 매커니즘이 적용됩니다.

<image src = https://user-images.githubusercontent.com/48677363/109592413-9444be80-7b52-11eb-9e93-9931e4c4f086.png width = 500>

각 층에서 정점 𝑖로부터 이웃 𝑗로의 가중치 $a_{𝒊𝒋}$는 3 Step을 통해 계산됩니다.

(1) 해당 층의 정점 𝑖의 임베딩 $h_𝑖$에 신경망 𝑾를 곱해 새로운 임베딩($h_iW$)을 얻습니다.
$$\tilde{h_{i}} = h_iW$$

(2) 정점 𝑖와 정점 𝑗의 새로운 임베딩을 concat한 후, 어텐션 계수 𝒂를 내적합니다 어텐션 계수 𝒂는 모든 정점이 공유하는 학습 변수입니다.
$$e_{ij} = a^T[concat(\tilde{h_{i}} \tilde{h_{j}})]$$

(3) 소프트맥스(Softmax)를 취함으로써 각 이웃 정점의 가중치를 확률값으로 변환해줍니다.

해당 과정에서도 **Multi-head Attention**을 적용함으로써 동시에 학습하여 사용할 수 있습니다. k개의 head로 얻게 된 value를 concat 후 집계하여 하나의 attention value로 사용함으로써 보다 풍부한 표현이 가능하게 됩니다. 아래 그림에서는 3개의 head로 구현되어 있음을 알 수 있습니다.

<image src = https://user-images.githubusercontent.com/48677363/109939344-50011c00-7d14-11eb-8df1-ea8f9cc98659.png width = 500>

결과적으로 GNN과 Attention이 결합된 GAT는 정점 분류의 정확도가 향상됨을 보였습니다. 확실히 이웃 정점의 정보를 단순 집계하는 것보다 해당 노드와의 관계를 고려한 Attention 매커니즘이 보다 표현력있음으로 이해하게 되었습니다.

<image src = https://user-images.githubusercontent.com/48677363/109939780-bb4aee00-7d14-11eb-9a2b-6b2d62535138.png width = 500>

#### 2) 그래프 표현 학습과 그래프 풀링

그래프 표현 학습은 **그래프 전체를 벡터의 형태로 표현**하는 것으로 그래프 임베딩이라고도 부릅니다.

개별 정점을 벡터의 형태로 표현하는 것은 이전 강의에서 배웠던 정점 표현 학습, Node Embedding이며 그래프 임베딩은 벡터의 형태로 표현된 그래프 자체를 의미합니다. 그래프 임베딩은 그래프 분류 등에 활용됩니다.

그래프 풀링, Graph Pooling이란 **정점 임베딩으로부터 그래프 임베딩을 얻는 과정**을 말합니다. 평균 등 단순한 방법보다 그래프의 구조를 고려한 방법을 사용할 경우 그래프 분류 등의 후속 과제에서 보다 높은 성능을 얻을 수 있는 것으로 알려져 있습니다. 아래 그림은 군집 구조를 활용하여 최종 그래프 임베딩을 계산하고 분류 문제에 활용한 것을 나타냅니다. 

<image src = https://user-images.githubusercontent.com/48677363/109943670-c738af00-7d18-11eb-9635-0692a09b5895.png width = 500>


#### 3) 지나친 획일화 문제, Over-smoothing

지나친 획일화 문제란 **그래프 신경망의 층이 깊어지면서 정점의 임베딩이 서로 유사해지는 현상**을 의미합니다.

지나친 획일화 문제는 앞선 강의에서 다룬 작은 세상 효과와 관련이 있습니다. GNN에서 layer의 개수는 이웃의 이웃을 보게 되는 깊이를 의미합니다. 즉, 1-layer는 이웃의 정보로부터 임베딩을 얻게 되며 5-layer는 이웃^5의 깊이로부터 임베딩을 얻게 됨을 말합니다. 이웃의 깊이가 깊어질수록 정점들이 참고하는 이웃들의 정보가 비슷해질 수 있기 때문에 결과적으로 유사한 임베딩 결과를 얻게 됩니다. 

![image](https://user-images.githubusercontent.com/48677363/109946190-50e97c00-7d1b-11eb-8de5-3893dc7452ea.png)

지나친 획일화의 결과로 그래프 신경망의 층의 수를 늘렸을 때, 후속 과제에서의 정확도가 감소하는 현상이 발견되었습니다. 
