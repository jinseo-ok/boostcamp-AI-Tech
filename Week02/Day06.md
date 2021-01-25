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

<img src = https://user-images.githubusercontent.com/48677363/105653267-b5384500-5efe-11eb-9640-c1cbb2e6f97f.png width = 600>

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

  - all & any

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


```