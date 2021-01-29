# Week02 - AI Math

## [Day 09] - Pandas II / 확률론 맛보기

### 1. Pandas II

##### (4) 데이터 연산

pandas에는 데이터 연산 및 집계를 위해 제공하는 다양한 함수가 있다.

다양한 함수를 실제로 적용해보면서 실습하는 것도 중요하지만 [pandas 공식 홈페이지](https://pandas.pydata.org/pandas-docs/stable/index.html)에서 제공하는 해당 함수의 소스코드와 원리를 보면서 보다 깊이 이해하는 것도 중요하다고 생각한다.

##### groupby [[ref]](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)  [[source]](https://github.com/pandas-dev/pandas/blob/v1.2.1/pandas/core/frame.py#L6600-L6725)

groupby 연산은 by (기준)에 따라 데이터프레임을 split하여 기준끼리 combination한 다음에 function을 적용하고 결과를 combining 한다.

  - 기준은 보통 column에 존재하는 categorical한 고유값으로 선택함
  - 다수의 columns에서 나타나는 unique 조합을 기준으로 선택할 수 있음
  - 특정 기준으로 그룹을 형성하게 되면 그룹 내에 적용할 수 있는 3가지 유형의 연산이 있음
    - Aggregation: 요약된 통계정보를 추출
    - Transformation: 정보 변경
    - Filtration: 정보 필터링

##### Pivot Table & Crosstab [[ref]](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html?highlight=pivot%20table#pandas.pivot_table)  [[source]](https://github.com/pandas-dev/pandas/blob/v1.2.1/pandas/core/reshape/pivot.py#L37-L200)

grouping된 데이터를 스프레드시트 형태로 보고 싶을 때 사용할 수 있는 함수이다. 즉, 행과 열을 기준으로 grouping된 데이터이며, 행과 열을 기준으로 groupby 후 unstack한 결과와 똑같다.

##### Merge & Concat

2개의 데이터를 병합할 때, 사용하는 함수이다.

데이터를 병합할 때, join과 같이 key값을 기준으로 데이터를 병합할 때는 `merge` 를 사용하고 단순 행과 열로 데이터를 합칠 때는 `concat`을 사용한다.

  - **Merge** [[ref]](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html?highlight=merge#pandas.merge)    [[source]](https://github.com/pandas-dev/pandas/blob/v1.2.1/pandas/core/reshape/merge.py#L57-L89)

merge로 데이터를 합칠 때에는, key를 기준으로 데이터를 병합하게 된다. 이 때, key에 대한 row의 개수를 유념하면서 병합을 시도해야한다. 

<image src = https://user-images.githubusercontent.com/48677363/106106490-f5aaf380-6188-11eb-9044-2ab13b19fb4e.png width = 500>


  - **concat** [[ref]](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html?highlight=concat)    [[source]](https://github.com/pandas-dev/pandas/blob/v1.2.1/pandas/core/reshape/concat.py#L82-L298)

concat은 단순히 데이터를 합칠 때 사용한다. 축을 기준으로 데이터를 병합하기 때문에 데이터 사이즈와 칼럼, 인덱스를 고려하면서 병합을 시도해야한다.

<image src = https://user-images.githubusercontent.com/48677363/106113740-27748800-6192-11eb-9122-b57585dd9e7f.png width = 600>


-----------

### 2. 확률론 맛보기

확률(Probability)은 해당 사건(event)이 발생할 가능성을 의미한다. 분류 문제에 있어 아주 단순하게, '입력값 $X$가 $Y$라는 결과로 나올 확률'이라고 확률론에 기반하여 딥러닝 문제를 해석할 수 있다.

딥러닝은 확률론 기반의 기계학습 이론에 바탕을 두고 있다. 기계학습에서 사용되는 손실함수(loss function)들의 작동 원리는 데이터 공간을 통계적으로 해석해서 유도하게 된다.

예를 들어, 회귀 분석에서 손실함수로 사용되는 $L2-norm$은 예측오차의 분산을 가장 최소화하는 방향으로 학습하도록 유도한다. 또한 분류 문제에서 사용되는 교차 엔트로피는 모델 예측의 불확실성을 최소화하는 방향으로 학습하도록 유도한다.



#### 1) 확률 변수 (Random Variable) [[ref]](https://devkihyun.github.io/study/Machine-learining-and-Probability/)

확률 변수는 발생가능한 모든 사건(event)의 집합인 표본공간에 존재하는 특정한 사건이 발생할 확률을 대응시켜주는 함수이다.
확률 변수는 $X$로 표기하고, 확률 변수가 취할 수 있는 실수값을 $x$, 그리고 $x$의 집합을 상태공간이라고 한다.

동전 던지기로 확률 변수에 대해서 보다 쉽게 이해할 수 있다.

먼저 동전을 2번 던지게 되면 ${(앞면, 앞면), (뒷면, 앞면), (앞면, 뒷면), (뒷면, 뒷면)}$ 4개의 사건이 존재하는 포본공간이 만들어지게 된다. 이 때, 앞면이 나오는 횟수라는 확률 변수의 사건의 수를 정리해보자면, ${(앞면 0 : 1), (앞면 1 : 2), (앞면 2 : 1)}$가 되며, 확률 변수가 취할 수 있는 값 x와 상태공간은 앞면이 나오는 횟수 자체에 대한 경우의 수인 ${앞면 0, 앞면 1, 앞면 2}$가 된다.

해당 예시를 수식으로 표현하자면, 확률변수 $X$에서 특정 $x$가 나올 확률로 대응 해주는 확률 함수 P를 사용하여 다음과 같이 표현할 수 있다.

  - $P(X = 0) = 1/4$
  - $P(X = 1) = 1/2$
  - $P(X = 2) = 1/4$

예시로 들었던 동전이 앞면이 나올 확률 변수의 상태공간 값인 0,1,2는 이산적(discrete)이기 때문에 확률변수 $X$는 이산 확률 변수(discrete random variable)이다. 만약 확률 변수 $X$가 연속적(continuous)이면 연속 확률 변수(continuous random variable)이다.

  - 이산형 확률 변수는 확률 변수가 가질 수 있는 경우의 수를 모두 고려하여 확률을 더해서 모델링함
  - 연속형 확률 변수는 데이터 공간에 정의된 확률 변수의 밀도(density) 위에서의 적분을 통해 모델링함

#### 2) 확률 분포 (Probability Distribution)

확률 분포는 확률 변수 $X$가 취할 수 있는 모든 $x$와 그에 대한 확률값의 분포를 의미한다. 위에서 언급한 확률 변수의 종류에 따라서 확률 분포의 모델링의 접근법이 달라지기 때문에 주의하여야 한다. 

#### 3) 조건부 확률 (Conditional Probability)

조건부 확률은 **어떤 사건이 발생한 조건이 있을 경우, 다른 사건이 발생할 확률을 의미한다.**
제공되는(given) 사건이 $A$라고 하고 우리가 구하고자 하는 사건 $B$에 대한 조건부 확률을 기호로 나타내면 $P(B∣A)$ 이며, 원하는 결과는 $B$이고 그 단서로 사건 $A$를 고려하게 된다.

<image src = https://user-images.githubusercontent.com/48677363/106228757-5c82e800-622f-11eb-92a9-c7e4eadd6176.png width = 300>

조건부 확률을 벤다이어그램으로 간단하게 표현하게 되면 다음과 같다. 사건 A의 발생을 전제로 사건 B가 발생해야 하기 때문에 **b영역**에 해당하게 된다.

**기대값 (Expectation)**

기대값은 데이터를 대표하는 통계량이면서 동시에 확률분포를 통해 다른 통계적 범함수를 계산하는데 사용된다. 기대값을 이용해 분산, 첨도, 공분산 등 여러 통계량을 계산할 수 있다.


**몬테카를로 샘플링 (Monte Carlo)**

기계학습의 대부분의 경우에는 확률분포를 명시적으로 알 수 없기 때문에, 데이터를 이용하여 기대값을 계산하기 위해 몬테카를로 샘플링 방법을 사용한다.

몬테카를로 샘플링은 이산형 혹은 연속형이든 상관없이 성립하지만 독립 추출만 보장된다면 대수의 법치에 의해 수렴성을 보장할 수 있다.

