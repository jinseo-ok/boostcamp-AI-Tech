# Week05 - Graph

## [Day 21] - 그래프 이론 기초 & 그래프 패턴

### 1. 그래프란 무엇이고 왜 중요한가?

#### 1) 그래프

그래프, Graph는 **정점 집합** 과 **간선 집합** 으로 이루어진 수학적 구조입니다. 이 때, 그래프를 네트워크(Network)라고도 하며, 정점을 부르는 말로는 **노드(node), vertex, 정점, 꼭지점** 등이 있고, 간선을 부르는 말로 **엣지(edge), 간선, 변, 링크**라고도 합니다.

이러한 정점의 집합을 $V$, 간선의 집합을 $E$, 그래프를 $G = (V, E)$라고 합니다. 간선과 정점의 집합으로 이루어진 그래프는 하나의 간선의 두 개의 정점을 연결되어야 하며, 모든 정점 쌍이 반드시 간선으로 직접 연결되지는 않다는 특징을 가지고 있습니다.

<center>
<image src = https://user-images.githubusercontent.com/48677363/108649678-26b8e280-7501-11eb-84a9-17c7763bf293.png width = 400>
</center>

#### 2) 그래프의 중요성

그래프가 중요한 이유는 인간 세계에서 발견되는 **복잡계, Complex System**과 유사한 점이 정말 많기 때문입니다. 예를 들어, 사회는 70억 인구로 구성된 복잡계이며 통신 시스템 또한 전자 장치로 구성된 복잡계, 그 밖에도 정보와 지식, 뇌, 신체 등 역시 복잡계로 생각할 수 있습니다.

다양한 분야에서 발생하는 복잡계에서 관측되는 공통적인 특징은 **구성 요소 간의 복잡한 상호작용**입니다. 이러한 복잡한 상호작용을 표현하기 위해 수학적 구조인 그래프가 사용될 수 있습니다.

즉, 그래프는 **복잡계를 효과적으로 표현하고 분석하기 위한 언어**입니다.

<image src = https://user-images.githubusercontent.com/48677363/108653518-f9246700-7509-11eb-90f2-7c30e7dd443a.png width = 700>

<br>

복잡계를 이해하고, 복잡계에 대한 정확한 예측을 하기 위해서는 복잡계 이면에 있는 그래프에 대한 이해가 반드시 필요합니다. 그래프에 대한 기본적인 개념에 대해 이해하게 된다면 다양한 분야에 존재하는 복잡계를 이해하고 활용할 수 있습니다.

#### 3) 그래프 관련 과제

**정점 분류 문제, Node Classification**

정점이 여러 유형을 가진 경유, 각 정점의 유형을 예측하는 과제입니다.

- 트위터에서 공유(Retweet) 관계(간선)를 분석하여, 각 사용자(정점)의 정치적 성향을 뷴류
- 단백질의 상호작용을 분석하여 단백질의 역할(Role)를 분류

**연결 예측 문제, Link Prediction**

정점과 정점이 연결에 대해 다루는 과제입니다.

- 거시적 관점: 주어진 그래프가 어떻게 성장할지 예측하는 과제
- 미시적 관점: 각 정점이 어떤 정점과 연결될지 예측하는 과제, 추천 문제와 연관성이 높음

**군집 분석 문제, Community Detection**

연결 관계로부터 사회적 무리(Social Circle)을 찾아 군집하는 과제입니다. 주어진 그래프에서 정점과 간선의 분포 및 관계를 기반으로 특징을 찾고 군집을 형성하게 됩니다.

**랭킹 및 정보 검색 문제, Ranking and Information Retrieval**

웹이라는 거대한 그래프로부터 중요하고 관련된 웹페이지를 찾아내는 과제입니다.

**정보 전파 및 바이럴 마케팅 문제, Information Cascading and Viral Marketing**

그래프를 통해 정보가 전달되는 과정 그리고 정보 전달의 최대화를 분석하는 과제입니다.

#### 4) 그래프 기본 용어

##### (1) 그래프 유형 및 분류

- Undirected Graph, 무방향 그래프
  -  정점과 정점 사이의 간선에 방향이 없는 그래프
  -  페이스북 친구 그래프

- Directed Graph, 방향 그래프
  - 정점과 정점 사이의 간선의 방향이 존재하는 그래프
  - 인용 그래프, 트위터 팔로우 그래프(팔로잉 과 팔로워 존재)

- Unweighted Graph, 
  - 간선이 모두 동일한 의미를 가지게 되는 그래프
  - 웹 그래프, 페이스북 친구 그래프


- Weighted Graph, 가중치 그래프
  - 간선의 숫자를 부여함으로써 간선 사이에도 차별을 둘 수 있는 그래프
  - 전화 그래프, 유사도 그래프

- Unpartite Graph, 동종 그래프
  - 단일 종류의 정점을 가지는 그래프
  - 웹 그래프, 페이스북 친구 그래프

- Bipartitle Graph, 이종 그래프
  - 두 종류의 정점을 가지는 그래프
  - 다른 종류의 정점 사이에만 간설이 발생함
  - 전자 상거래 구매내역(사용자-상품), 영화 출연 그래프(배우-영화)

##### (2) Neighbor

정점의 **이웃, Neighbor**은 그 정점과 연결된 다른 정점을 의미합니다. 정점 $v$의 이웃들의 집합을 보통 $N_{v}$ 혹은 N_{(v)}로 지칭합니다. 

방향성이 있는 그래프에서는 간선의 방향에 따라 연결된 정점을 구분하게 됩니다.

- 정점 v에서 간선이 나가는 이웃, Out-Neighbor의 집합을 보통 $N_{out}(V)$로 지칭함
- 정점 v로 간선이 들어오는 이웃, In-Neighbor의 집합을 보통 $N_{in}(V)$로 지칭함

<center>
<image src = https://user-images.githubusercontent.com/48677363/108671244-7b6d5500-7523-11eb-9a72-ccc18b18d544.png width = 500>
</center>

##### (3) **경로, Path**

- u 에서 시작해서 v 에서 끝나야 함
- 순열에서 연속된 정점은 간선으로 연결되어 있어야 함
- 위 두가지 조건을 만족하는 정점들의 순열(Sequence)

##### (4) **거리, Distance**

- 해당 경로 상에 놓이는 간선의 수로 정의됨
- 두 정점 사이의 거리는 최단 경로를 의미함

##### (5)**지름, Diameter**

- 정점 간 거리의 최댓값임

##### (6) 연결성, Degree

정점의 연결성, Degree는 해당 정점과 연결된 간선의 수를 의미합니다. 특정 정점의 연결성은 해당 정점의 이웃들의 수와 같다고 할 수 있습니다. 보통 정점 v의 연결성은 $d(v), d_{v}$ 혹은 $|N(v)|$라고 지칭합니다.

방향 그래프의 경우에는, 들어오고 나가는 연결성을 구분하여 간선의 수를 계산합니다.

##### (7) 연결 요소, Connected Component

- 연결 요소에 속하는 정점들은 경로로 연결될 수 없음
- 위 조건을 만족하면서 정점을 추가할 수 없음


#### 5) 그래프 표현 및 저장

그래프를 표현하기 위해서 항상 그림으로 나타낼 수 없기 때문에 컴퓨터가 이해할 수 있는 방법으로 저장 및 표현이 이뤄져야 합니다.

**간선 리스트, Edge List**

그래프를 간선들의 리스트로 저장합니다. 각 간선은 해당 간선이 연결하는 두 정점들의 순서쌍, pair로 저장됩니다. 이 때, 방향이 존재하는 방향 그래프일 경우에는 (출발점, 도착점) 순서로 저장할 수 있습니다.

<image src = https://user-images.githubusercontent.com/48677363/108683545-17ec2300-7535-11eb-97d4-0906f97315c9.png width = 250>

**인접 리스트, Adjacent List**

각 정점의 이웃들을 리스트로 저장합니다. 이 때에도, 방향 그래프일 경우, 나가고 들어오는 이웃의 차이가 있기 때문에 다르게 pair를 묶을 수 있습니다.

<image src = https://user-images.githubusercontent.com/48677363/108683771-6994ad80-7535-11eb-9f15-6e1bf8434bdb.png width = 400>

**인접 행렬, Adjacency Matrix**

인접 리스트를 바탕으로 정점 * 정점의 간선을 2차원 행렬로 표현합니다. 이 때, 무방향 그래프일 경우 인접 행렬은 대칭을 이룰 것이고, 방향 그래프일 경우 인접 행렬은 대칭을 이루지 않습니다.
 
<image src = https://user-images.githubusercontent.com/48677363/108684285-0f481c80-7536-11eb-9537-282dfa978e03.png width = 600>

---------

### 2. 실제 그래프는 어떻게 생겼나?

이번 강의에서는 그래프가 가지는 다양한 패턴들에 대해서 다룹니다.

그래프는 다양한 복잡계의 정보를 담고 있습니다. 예를 들어, 페이스북 네트워크에서는 사람을 정점으로 나타내고, 친구 관계를 간선으로 표현할 수 있습니다. 이러한 실제 그래프들은 랜덤으로 생성한 그래프 모댈과는 다른 패턴들을 보이게 되며, 우리는 실제 그래프의 패턴들을 활용하여 효과적으로 그래프를 분석할 수 있습니다. 실제 그래프와 랜덤 그래프의 차이를 바탕으로 작은 세상 효과, 연결성의 두터운 꼬리 분포, 군집 구조 등 실제 그래프의 다양한 패턴들을 학습할 수 있습니다. 

#### 1) 실제 그래프(Real Graph) vs 랜덤 그래프(Random Graph)

**실제 그래프, Real Graph**

실제 그래프란 다양한 복잡계인 실제 데이터로부터 얻어진 그래프를 의미합니다. 

**랜덤 그래프, Random Graph**

실제 그래프와의 비교를 통해 차이를 분석하기 위한 확률적 과정을 통해 생성한 그래프를 의미합니다.

가장 단순한 형태의 랜덤 그래프 모델로는 에르되스와 레니가 제안한 랜덤 그래프 모델이 있습니다. 에르되스-레니 랜덤 그래프는 임의의 두 정점 사이에 간선이 존재하는지 여부를 동일한 확률 분포에 의해 결정한다는 특성이 있습니다.

에르되스-레니 랜던그래프 $G(n, p)$ 는 

- n 개의 정점을 가짐
- 임의의 두 개의 정점 사이에 간선이 존재할 확률을 p 임
- 정점 간의 연결은 서로 독립적임

**Q. $G(3, 0.3)$ 에 의해 생성될 수 있는 그래프와 각각의 확률은?** 

<image src = https://user-images.githubusercontent.com/48677363/108689554-926c7100-753c-11eb-9849-342aa08ff7c0.png width = 500>

#### 2) 작은 세상 효과

작은 세상 효과는 실제 그래프의 임의의 두 정점 사이의 거리가 생각보다 작음을 의미하는 효과입니다.

<image src = https://user-images.githubusercontent.com/48677363/108691629-0b6cc800-753f-11eb-92b4-fbff02443e39.png width = 300>

하지만 모든 그래프 구조에서 작은 세상 효과가 존재하는 것은 아닙니다. 체인, 사이클, 격자 그래프에서는 작은 세상 효과가 존재하지 않습니다.

<image src = https://user-images.githubusercontent.com/48677363/108691744-2b9c8700-753f-11eb-87ce-b2ee6f1dc404.png width = 500>

#### 3) 연결성의 두터운 꼬리 분포

실제 그래프의 연결성 분포는 두터운 꼬리를 갖습니다. 즉, 연결성이 매우 높은 허브(Hub) 정점이 존재함을 의미합니다. 반면 랜덤 그래프의 연결성 분포는 높은 확률로 정규분포와 유사합니다. 

<image src = https://user-images.githubusercontent.com/48677363/108692779-6521c200-7540-11eb-9c10-8fcadb606ff3.png>

#### 3) 거대 연결 요소

실제 그래프에서는 거대 연결 요소(Giant Connected Component)가 존재합니다. 거래 연결 요소는 대다수의 정점을 포함합니다. 예를 들어, MSN 메신저 그래프에서는 99.9%의 정점이 하나의 거대 연결 요소에 포함되는 현상이 발견됩니다. 

놀랍게도 랜덤 그래프에서 높은 확률로 거대 연결 요소가 존재합니다. 단 정점들의 평균 연결성이 1보다 충분히 커야 거대 연결 요소 현상이 발견됩니다.

![image](https://user-images.githubusercontent.com/48677363/108695009-217c8780-7543-11eb-85d3-07440da426ea.png)

#### 4) 군집 구조

군집은 정점들의 집합으로 다음의 조건을 만족합니다.

- 집합에 속하는 정점 사이에는 많은 간선이 존재함
- 집합에 속하는 정점과 그렇지 않은 정점 사이에는 적은 수의 간선이 존재함

**지역적 군집 계수(Local Clustering Coefficient)**

정점별로 군집의 형성 정도를 의미하는 지역적 군집 계수(Local Clustering Coefficient)를 정의할 수 있습니다. 정점 v의 지역적 군집 계수는 정점 v의 이웃 쌍 중 간선으로 연결된 쌍의 비율을 의미합니다.

첫번째 그래프의 경우, 정점 1과 연결된 이웃은 {2,3,4,5}입니다. 이웃의 쌍은 (2,3), (3,4), (4,5), (2,4), (2,5), (3,5)로 총 6개이며 이 중 간선으로 연결된 쌍은 (2,3), (2,4), (3,5), (4,5)입니다. 정점 1의 지역적 군집 계수는 4/6으로 0.66에 가깝습니다.

<image src = https://user-images.githubusercontent.com/48677363/108695701-f21a4a80-7543-11eb-9799-8d1420cf2797.png width = 500>

특정 정점과 이웃 정점의 지역적 군집 계수가 높게 되면 높은 확률로 군집을 형성함을 의미합니다. 

**전역 군집 계수(Global Clustering Coefficient)**

전체 그래프의 군집 형성 정도를 측정하기 위해 전역 군집 계수를 정의할 수 있습니다. 그래프 G의 전역 군집 계수는 각 정점에서의 지역적 군집 계수의 평균입니다.

실제 그래프는 대부분 군집 계수가 높으며, 많은 군집이 존재합니다. 군집이 존재하는 이유는 다음과 같습니다.

- 동질성(Homophily): 서로 유사한 정점끼리 간선으로 연결될 가능성이 높음
- 전이성(Transitivity): 공통 이웃이 있는 경우, 공통 이웃이 매개 역할을 해줄 수 있음

반면 랜덤 그래프에서는 지역적 혹은 전역 군집 계수가 높지 않습니다. 구체적으로 랜덤 그래프 $G(n, p)$ 에서의 군집 계수는 p 입니다. 랜덤 그래프에서의 간선 연결은 독립적이기 때문에 공통 이웃의 존재 여부가 간선ㅁ 여결 확률에 영향을 미치지 않으므로 동질성 및 전이성에 대한 효과가 발휘되지 않습니다.
