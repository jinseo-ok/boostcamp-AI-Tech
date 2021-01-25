# Week02 - AI Math

## [Day 06] - Numpy/벡터/행렬

### 1. Numpy

numpy는 python에서 진행되는 모든 데이터 분석과 AI 엔지니어링에 있어 가장 기초적이며 필수적으로 사용되는 패키지이다. Numerical Python의 약자로, 일반적으로 과학 계산에서 많이 사용하는 선형대수의 계산식을 python에서 구현할 수 있도록 도와준다.

numpy는 패키지 자체의 기능으로도 많이 사용되지만 이후에 사용되는 scipy나 pandas의 base 객체로도 사용되며, numpy의 특유 문법이 그대로 딥러닝 프레임워크인 pytorch나 tensorflow에서도 사용되기 때문에 numpy를 숙지하는 것은 매우 중요하다.

#### 1) Numpy part I

##### (1) 특징

  - python의 고성능 과학 계산용 패키지
  - Matrix와 vector와 같은 array 연산의 표준
  - 일반 List에 연산이 빠르고, 메모리 부분에서도 효율적
  - 반복문 없이 배열을 활용한 연산 처리가 가능함
  - 선형대수와 관련된 기능을 제공함
  - C, C+, 포트란 등의 언어와 통합 가능

##### (2) ndarray(numpy dimension array)

  - numpy는 np.array 함수를 활용하여 ndarray라는 객체 배열을 생성함
  - numpy는 하나의 데이터 type만 배열에 포함할 수 있음
  - List와 가장 큰 차이점은 **dynamic typing not supported**

```python
import numpy as np
arr = np.array(['1', '2', 3.0, 4, 5.01], float)
type(arr)
-> numpy.ndarray
type(arr[0]) # 모두 정의된 float으로 변환
-> numpy.float64
```

##### (3) array creation

  - numpy의 경우 값 자체가 순서대로 메모리에 저장됨
  - list의 경우 값을 참고하는 메모리 주소값을 거쳐 생성됨
  - 그러므로, numpy는 값 자체의 연산이 굉장이 빠르지만 list는 메모리 주소를 통해 값을 연산해야하기 때문에 보다 느림
  - 또한, 값을 저장하는 메모리의 크기가 일정하기 때문에 메모리 공간을 효율적이게 관리할 수 있음

<img src = https://user-images.githubusercontent.com/48677363/105651314-dea2a200-5ef9-11eb-9b7d-ae597b49bdad.png width = 600>

```python
# List
a = [1,2,3,4,5]
b = [5,4,3,2,1]
a is b
-> False
a[0] is b[-1] # 1 이라는 값을 참고하는 메모리 주소가 같기 때문에
-> True

# Array
a = np.array(a)
b = np.array(b)
a[0] is b[-1] # 값이 다른 메모리에 저장되어 있기 때문에
-> False
```

  - array의 RANK에 따라 불리는 이름이 있음, 선형대수에서 값을 표현하는 방법

| Rank |  Name |    Example    |   
|:----:|:-----:|:-------------:|
|  0  | scalar | `7` |
|  1  | vector | `[10, 10]` |
|  2  | matrix | `[[10, 10], [15, 15]]`|
|  3  | 3-tensor | `[[[1, 5, 9], [2,6,10]], [[3, 7, 11], [4, 8, 12]]]` |
|  n  | n-tensor | |

  - ndarray의 single element가 가지는 data type
  - 각 element가 차지하는 memory의 크기가 결정됨
  - C 언어의 data type과 compatible함
  - nbytes: ndarray object의 메모리 크기를 반환함

```python
np.array([[1,2,3], [4.5, 5, 6]], dtype = np.float32).nbytes
-> 24 (6 * 4bytes)

np.array([[1,2,3], [4.5, 5, 6]], dtype = np.int8).nbytes
-> 6 (6 * 1bytes)

np.array([[1,2,3], [4.5, 5, 6]], dtype = np.np.float64).nbytes
-> 48 (6 * 8bytes)
```

<img src = https://user-images.githubusercontent.com/48677363/105653267-b5384500-5efe-11eb-9640-c1cbb2e6f97f.png width = 800>

#### 2) Numpy part II

##### (1) Handling shape

  - **reshape**, array의 element의 개수가 동일한 상태에서 shape의 size를 변경하는 방법

    - -1를 기반으로 본래 element의 개수에 맞추어 자동 reshape

```python
np.array([[1,2,3,4], [5,6,7,8]]).shape
-> (2, 4)

np.array([[1,2,3,4], [5,6,7,8]]).reshape(4, 2).shape
-> (4, 2)

np.array([[1,2,3,4], [5,6,7,8]]).reshape(-1, 1).shape
-> (8, 1)

np.array([[1,2,3,4], [5,6,7,8]]).reshape(1, -1, 2).shape
-> (1, 4, 2)
```

  - **flatten**, 다차원 array를 1차원 array로 변환

```python
np.array([[1,2,3,4], [5,6,7,8]]).flatten().shape
-> (8, )
```

##### (2) indexing & slicing

  - **indexing**
    - `arr[행][열]` 로 해당 값을 가져올 수 있음
    - 추가적으로 새롭게 값을 할당하여 변경 가능함

  - **slicing**
    - list와 다르게, 행과 열 부분을 나눠서 slicing이 가능함
    - matrix의 부분 집합을 추출할 때 유용함

```python
arr = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12], [13, 14, 15, 16]])
-> array([[ 1,  2,  3,  4],
          [ 5,  6,  7,  8],
          [ 9, 10, 11, 12],
          [13, 14, 15, 16]])

arr[:, 2:] # 전체 행의 2열 이상
-> array([[ 3,  4],
          [ 7,  8],
          [11, 12],
          [15, 16]])

arr[1, 1:3] # 1행의 1~2열
-> array([6, 7])

arr[1:3] # 1~2행
-> array([[ 5,  6,  7,  8],
          [ 9, 10, 11, 12]])

arr[:, 1:2] # 1~2열
-> array([[ 2,  3],
          [ 6,  7],
          [10, 11],
          [14, 15]])

arr[:, ::2] # 시작부터 2열씩
-> array([[ 1,  3],
          [ 5,  7],
          [ 9, 11],
          [13, 15]])

arr[::2, ::3] # 시작부터 행은 2행씩, 열은 3열씩 겹치는 부분
-> array([[ 1,  4],
          [ 9, 12]])
```

##### (3) creation functions

  - **arange**, array의 범위를 지정하여, 값의 list를 생성하는 명령어
  - **zeros**, 0으로만 shape 크기로 ndarray 생성하는 명령어
  - **ones**, 1로만 shape 크기로 ndarray 생성하는 명령어
  - **empty**, shape 크기로 빈 ndarray 생성하는 명령어(memory initialize 되지는 않음)
  - **{}_like**, 지정한 ndarray의 shape 만큼 1, 0 또는 empty array를 반환
    - empty_like: Return an empty array with shape and type of input.
    - zeros_like: Return an array of zeros with shape and type of input.
    - full_like: Return a new array with shape of input filled with value.
    - ones_like: Return a new array setting values to one.
  - **identity**, 단위 행렬을 생성함(n = number of rows)
  - **eye**, 단위 행렬을 생성하되, 시작 지점 및 행렬 크기 지정 가능
  - **diag**, 대각 행렬의 값을 추출함(k = start index)
  - **random**, 데이터 분포에 따른 sampling을 통해 ndarray 생성하는 명령어
    - random.uniform(): 균등분포
    - random.normal(): 정규분포
    - random.exponential(): 지수분포
  
##### (4) operation functions

| function |  Description | function |  Description |
|:----:|:-----:|:----:|:-----:|
|  sum  | 합계 |  median | 중간값 |
|  mean  | 평균 |  unique | 고유값 |
|  std  | 표준편차 |  cumsum | 누적값 |
|  min  | 최소값 | corrcoef | 상관계수 |
|  max | 최대값 |


  - axis: 모든 operation function을 실행할 때, 기준이 되는 dimenstion 축(default = 1(행))
  - **2차원 axis**

<img src = https://user-images.githubusercontent.com/48677363/105662440-0ce0ab80-5f13-11eb-9370-4f4e94b1333c.png width = 300>

  - **3차원 axis**

<img src = https://user-images.githubusercontent.com/48677363/105662486-2c77d400-5f13-11eb-9047-63e904da2fe2.png width = 350>

##### (5) concating shape

  - **concatenate**, ndarray를 합치는 명령어
    - vstack: concat(axis = 0)
    - hstack: concat(axis = 1)

```python
a = np.array([1,2,3])
b = np.array([2,3,4])
np.vstack((a, b))
-> array([[1,2,3], 
          [2,3,4]])

np.hstack((a,b))
-> array([1, 2, 3, 2, 3, 4])

np.concatenate((a,b), axis = 0)
-> array([1, 2, 3, 2, 3, 4])
```

  - **newaxis**, 하나의 축을 추가할 때 사용하는 명령어
```python
b = np.array([5,6])
b.shape
-> (2, )

b.reshape(-1, 2).shape
-> (1, 2)

b[np.newaxis, :].shape
-> (1, 2)
```

##### (6) array operation, 배열 연산

기본적으로 배열 간 사칙연산이 가능하다. 이 때, 사칙연산은 element-wise 연산이 적용된다.

  - **multiply**: 행렬의 element-wise 곱셈

```python
a = np.array([[1,2,3], [4,5,6]])
b = np.array([[1,2,3], [4,5,6]])

a * b
-> array([[ 1,  4,  9],
          [16, 25, 36]])

np.multiply(a,b)
-> array([[ 1,  4,  9],
          [16, 25, 36]])
```

  - **Dot product**: 행렬의 기본 연산

```python
a = np.array([[1,2,3], [4,5,6]])
b = np.array([[5,4], [6,5], [1,2]])

np.dot(a,b)
-> array([[20, 20],
          [56, 53]])

a.dot(b)
-> array([[20, 20],
          [56, 53]])
```

  - **transpose**: 전치 행렬로 변환

  - **broadcasting**: shape이 다른 배열 간 연산을 지원하는 기능, 하지만 모든 경우에 성립되지는 않음

```python
a = np.array([[1,2,3], [4,5,6]])
b = np.array([1,2])
a + b
-> ValueError: operands could not be broadcast together with shapes (2,3) (2,) 
```

  - **timeit**: jupyter 환경에서 코드의 퍼포먼스를 체크하는 함수


#### 3) Numpy part III

##### (1) comparisons

  - **all & any**

```python
arr = np.arange(10)
arr < 4
-> array([ True,  True,  True,  True, False, False, False, False, False, False])

np.any(a > 5) # 하나라도 조건에 만족한다면 True
-> True

np.all(arr > 5), np.all(arr < 10) # 모두 조건에 만족한다면 True, 그렇지 않다면 False
-> (False, True)
```

  - 배열의 shape이 동일할 때, element-wise comparision이 발생함

```python
a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3) # 두 조건의 and 조건
-> array([ True, False, False])

a = np.array([1, 3, 0], float)
np.logical_not(a < 2) # 기존 결과의 not 조건(역)
-> array([False,  True,  True])

a = np.array([1, 3, 0], float)
np.logical_or(a > 0 , a < 3)
-> array([ True,  True,  True])
```

  - **np.where**
    - `np.where(condition, [x, y])`: condition을 만족했을 때는 x, 그렇지 않을 때 y 반환
    - `np.where(condition)`: condition을 만족하는 index를 반환

  - **np.isnan**: None을 True로 반환, pandas의 isnull과 유사
  - **np.isfinite**: 무한대로 수렴하지 않는 값을 찾는 명령어

  - **argmax & argmin**: axis에 기반하여 array의 최대값 또는 최소값의 index를 반환하는 명령어
  - **argsort**: array를 정렬하여 본래의 index로 반환해주는 명령어

```python
arr = np.array([[1,2,4,7], [9, 88, 6, 45], [9, 76, 3, 4]])
-> array([[ 1,  2,  4,  7],
          [ 9, 88,  6, 45],
          [ 9, 76,  3,  4]])

np.argmax(arr, axis = 1)
-> array([3, 1, 1])

np.argmin(arr, axis = 0)
-> array([0, 0, 2, 2])

arr = np.array([1,2,4,5,8,78,23,3])
arr.argosrt()
-> array([0, 1, 7, 2, 3, 4, 6, 5])
```

##### (2) boolean & fancy index

  - **boolean index**: 특정 조건에 따른 값을 배열 형태로 추출, pandas의 loc[조건]과 유사

```python
arr = np.array([[1,2,4,7], [9, 88, 6, 45], [9, 76, 3, 4]])
arr[arr > 10]
-> array([88, 45, 76])
```

  - **fancy index**: index를 조건으로 값을 lookup하여 추출, pandas의 loc[index]와 유사
    - matrix 형태의 데이터도 indexing 가능

```python
# 2차원
arr = np.arange(1, 10, 0.5)
index = np.array([0,0,1,2,1,3,3,1,4])
arr[index] # index의 범위를 넘어가면 IndexError 발생
-> array([1. , 1. , 1.5, 2. , 1.5, 2.5, 2.5, 1.5, 3. ])

arr.take(index)
-> array([1. , 1. , 1.5, 2. , 1.5, 2.5, 2.5, 1.5, 3. ])

# matrix
arr = np.array([[1,2,4,7], [9, 88, 6, 45], [9, 76, 3, 4]])
index_x = np.array([0,0,1,2,1,1])
index_y = np.array([0,1,2,0,0,2])
arr[index_x, index_y]
-> array([1, 2, 6, 9, 9, 6])
```

##### (3) numpy data i/o

  - csv 저장

```python
data = np.loadtxt('./numpy.txt', delimiter = '\t')
np.savetxt('numpy.csv', data, delimiter = ',')
```

  - numpy object - npy

```python
np.save('npy_file', arr = data)
```

---------

### 2. 벡터가 뭐에요?

벡터의 기본 개념과 연산, 노름에 대해 소개한다. 또한 두 벡터 사이의 거리와 각도, 그리고 내적에 대해 설명한다.
벡터는 딥러닝에서 매우 중요한 선형대수학의 기본 단위이기 때문에 기본적인 개념을 확실하게 학습하는 것이 중요하다. 벡터를 단순히 숫자의 표현으로 이해하는 것이 이닌 공간에서 어떤 의미를 가지고 있는 지를 이해하는 것이 중요하다.

노름이나 내적 같은 개념 또한, 그 자체로 가지는 기하학적인 성질과 이것이 실제 머신러닝에서 어떻게 사용되는지를 고민하고 적용해보는 것이 벡터 공부의 시작이다.

#### 1) 벡터

벡터는 숫자를 원소로 가지는 리스트(list) 또는 배열(array)를 말한다. 세로로 나열되어 있는 벡터를 열벡터, 가로로 나열되어 있는 벡터를 행벡터로 부른다.

<image src = https://user-images.githubusercontent.com/48677363/105707999-23a8f180-5f57-11eb-82fa-c3c52f405181.png width = 400>

  - 벡터는 공간에서 **한 점**을 나타냄
  - 벡터는 원점으로부터 상대적 위치와 방향을 의미함
  - 벡터에 숫자를 곱해주면 방향은 동일한 상태에서 길이만 변하게 됨(스칼라곱)
  - 벡터끼리 같은 모양을 가지면 벡터 간 덧셈 및 뺄셈과 곱셉(element-wise product)을 계산할 수 있음

❗스칼라곱: 스칼라가 1보다 크면 길이가 늘어나고, 1보다 작으면 길이가 줄어듬, 단 0보다 작으면 반대 방향으로 변함

<image src = https://user-images.githubusercontent.com/48677363/105708177-65399c80-5f57-11eb-8a54-466ce487f8cc.png width = 550> 

  - 두 벡터의 덧셈은 다른 벡터로부터 상대적 위치이동을 표현할 수 있음

<image src = https://user-images.githubusercontent.com/48677363/105708908-63bca400-5f58-11eb-9bf4-8a52655f79e2.png width = 700>


#### 2) 벡터의 노름

벡터의 노름(norm)은 원점에서부터의 거리를 말한다. 한 공간 사이에 표현되어 있는 벡터와 원점과의 거리를 의미하는데, 표현된 벡터와 원점과의 거리를 구하는 노름의 방법이 다양하다.
보통 l1, l2 노름 방법이 존재한다.

##### (1) L1 norm: 각 성분의 변화량의 절대값을 모두 더하여 계산하는 방법

##### (2) L2 norm: 피타고라스 정리를 이용해 유클리드 거리로 계산하는 방법

```python
def l1_norm(x):
    x_norm = np.abs(x)
    x_norm = np.sum(x_norm)
    return x_norm

def l2_norm(x):
    x_norm = x * x
    x_norm = np.sum(x_norm)
    x_norm = np.sqrt(x_norm)
    return x_norm
```

노름을 계산하는 방법이 다른 이유는 노름의 종류에 따라 **기하학적 성질**이 달라지기 때문이다. 머신러닝에선 각 성질들이 필요할 때가 다르므로 두 방법의 차이점을 인지하고 있어야 한다.

<image src = https://user-images.githubusercontent.com/48677363/105709782-87ccb500-5f59-11eb-8c12-3e199efb0aa8.png width = 500>

#### 3) 벡터간 거리 계산

L1, L2-노름을 이용해 두 벡터 사이의 거리를 계산할 수 있다. 노름의 방법에 따라 거리 계산 결과가 달라질 수 있다.

  - 두 벡터 사이의 거리를 이용하여 각도를 계산할 수 있음(제2 코사인 법칙)
  - 제2 코사인 법칙에서 분자는 두 벡터의 내적을 의미함

❗️[코사인 유사도](https://ko.wikipedia.org/wiki/코사인_유사도)(― 類似度, 영어: cosine similarity)는 내적공간의 두 벡터간 각도의 코사인값을 이용하여 측정된 벡터간의 유사한 정도를 의미한다.
  
```python
def angle(x, y):
    v = np.inner(x, y) / (l2_norm(x) * l2_norm(y))
    theta = np.arccos(v)
    return theta
```

  - 내적, inner-product는 정사영(orthogonal projection)된 벡터의 길이와 관련이 있음
    - $Proj(x)$의 길이는 코사인 법칙에 의해  $||x|| cos θ$ 가 됨
    - 내적은 두 벡터의 유사도를 측정하는데 보통 사용함

### 3. 행렬이 뭐에요?

행렬의 개념과 연산, 그리고 벡터공간에서 가지는 의미를 설명한다. 또한 연립방정식 풀기와 선형회귀분석에 응용하는 방법을 소개한다.
벡터의 확장된 개념인 행렬은 행(row)벡터를 원소로 가지는 2차원 배열로 벡터와 다르게 계산되는 연산들에 주의해야 한다. 행렬연산은 딥러닝에서 가장 핵심적인 연산이라고 볼 수 있을만큼 중요하고, 자주 사용되기 때문에 행렬 연산의 메커니즘, 그리고 이 때 가지는 기하학적 의미와 머신러닝에서 어떻게 사용되는지를 충분히 이해해야 한다.

#### 1) 행렬

행렬, matrix는 벡터를 원소로 가지는 2차원 배열을 말한다. 행렬은 n(행) x m(열)로 표현할 수 있다. 즉, 행렬은 행(row)과 열(column)이라는 인덱스를 가지게 된다.

  - 행렬의 특정 행(열)을 고정하면 행(열)벡터라 부름
  - 전치행렬(transpose matrix)은 행과 열의 인덱스가 바뀐 행렬을 의미함

<image src = https://user-images.githubusercontent.com/48677363/105719207-5eb22180-5f65-11eb-8991-ace786b99894.png width = 600>

  - 벡터가 공간에서 한 점을 의미한다면 행렬은 여러 점들을 의미함
  - 행렬의 행벡터 $x_{i}$ i번째 데이터를 의미함
  - 행렬의 $x_{ij}$는 i 번째 데이터의 j 번째 변수의 값을 의미함

<image src = https://user-images.githubusercontent.com/48677363/105719746-00397300-5f66-11eb-9853-b0ee88733fc0.png width = 600>

#### 2) 행렬의 연산

  - 행렬도 벡터와 같이 같은 모양을 가지면 덧셈 및 뺄셈이 가능함
  - 성분곱(스칼라곱) 또한 벡터와 똑같음, element-wise product

#### 3) 행렬 곱셈

행렬 곱셈은 벡터와 다른 방식으로 진행된다. 행렬 곱셈(matrix multiplication)은 $i$번째 행벡터와 $j$번째 열벡터 사이의 내적을 성분으로 가지는 행렬 계산을 의미한다.

  - 행렬 내적으로 `np.inner`를 사용함으로써, $i$번째 행벡터와 $j$번째 행벡터 사이의 내적을 계산할 수 있음
 
❗주의할 점은 수학에서 말하는 내적과는 다르므로 주의해야함!

<image src = https://user-images.githubusercontent.com/48677363/105720214-89e94080-5f66-11eb-8d93-0dc79f4731e9.png width = 600>

<br>
<br>

```python
a = np.array([[1,2,3], [4,5,6], [1,6,3]])
b = np.array([[1,2], [5,6], [8,1]])
np.dot(a, b)
-> array([[35, 17],
          [77, 44],
          [55, 41]])

np.inner(a, b) # a 배열의 행벡터와 b 배열의 행벡터의 사이즈가 일치하지 않기 때문에
-> ValueError: shapes (3,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)
```

