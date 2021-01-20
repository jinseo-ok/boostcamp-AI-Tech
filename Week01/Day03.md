# Week01 - Python

## [Day 03] - 파이썬 기초 문법 II

### 1. Python Data Structure

이번 강의에서는 python에 존재하는 자료구조에 대해 배운다. 자료구조란 데이터를 저장할 때, 데이터에 특징에 따라 효율적으로 저장하기 위한 데이터의 저장 및 표현 방식을 의미한다. 자료구조에 따라 데이터를 저장하는 방식과 표현 및 특징이 다르기 때문에 상황에 따라서 적합한 자료구조로 효율적인 개발이 가능하다.

**python 기본 데이터 구조**

  - 스택과 큐(stack & queue with list)
  - 튜플과 집합(tuple & set)
  - 사전(dictionary)
  - Collection 모듈

#### 1) Stack, 스택

  - 나중에 넣은 데이터를 먼저 반환하도록(Last In First Out, LIFO) 설계된 메모리 구조
  - Data의 입력을 push, 출력을 pop이라고 함
  - List를 사용해서 스택 구조를 구현 가능
  - append() 와 pop() 으 구현 가능


#### 2) Queue, 큐

  - 먼저 넣은 데이터를 먼저 반환하도록(First In First Out, FIFO) 설계된 메모리 구조
  - Stack과 대조되는 개념
  - List를 사용해서 큐 구조를 구현 가능
  - append() 와 pop(0) 으 구현 가능

#### 3) tuple, 튜플

  - **값의 변경이 불가능한 리스트**, immutable
  - 선언 시, '[]'가 아닌 '()'를 사용
  - List의 연산, 인덱싱, 슬라이싱 등을 동일하게 사용
  - 실수에 의한 변경을 사전에 방지하기 위해 튜플 선언
  - 값이 하나인 tuple은 반드시 ','를 붙여줘야 tuple로 인식됨

```
t = (1)
type(1)
-> int

t = (1, )
type(t)
-> tuple
```

#### 4) set, 집합

  - 값은 순서없이 저장(내장 순서가 고정되어 있음), 중복 불허 하는 자료형
  - set 객체 선언을 이용하여 객체 생성
  - 'add'와 'remove' 메소드를 통해 원소 추가와 삭제가 가능

| method |  Description  |   
|:----:|:-------------:|
|  add  | 1개의 원소 추가  |
|  reomve  |  1개의 원소 삭제   |  
|  update  |   다수의 원소 추가    |  
|  discard  |   1개의 원소 삭제   |  
|  clear |   모든 원소 삭제    | 

❗️'remove'와 'discard' 차이점: the remove() method will raise an error if the specified item does not exist, and the discard() method will not. [ref](https://www.w3schools.com/python/ref_set_discard.asp)

##### (1) 집합의 연산

  - 수학에서 활용되는 다양한 집합연산 가능

```
s1 = set([1,2,3,4,5])
s2 = set([3,4,5,6,7])

# 합집합
s1.union(s1)
s1 | s2
-> {1,2,3,4,5,6,7}

# 교집합
s1.intersection(s2)
s1 & s2
-> {3,4,5}

# 차집합
s1.difference(s2)
s1 - s2
-> {1,2,3}
```

#### 5) dictionary, 사전

  - 데이터 저장 시, key-value 형식으로 함께 저장하여 관리하는 자료형
  - 구분을 위한 데이터 고유 값을 indentifier 또는 Key 라고 함
  - Key를 활용하여 데이터 값(Value)를 관리함

#### 6) collectiions

  - List, Tuple, Dict에 대한 python built-in 확장 자료 구조(module)
  - 편의성, 실행 효율 등을 사용자에게 제공함

```
from collections import deque
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
```

##### (1) 모듈

  - deque
    - Stack과 Queue를 지원하는 모듈
    - List에 비해 효율적(빠른 자료 저장 방식을 지원함) 메모리 구조로 처리 속도 향상
    - rotate, reverse 등 Linked List의 특성을 지원함
    - 기존 list 형태의 함수를 모두 지원함

```
deque_list = deque()
for i in range(5):
  deque_list.append(i)
deque_list.appendleft(10)
-> deque([10, 0, 1, 2, 3, 4]) # appendleft로 왼쪽에 append됨, extendleft도 가능

deque_list.rotate(1)
-> deque([4, 10, 0, 1, 2, 3])
```

❗️ 해당 강의에서는 일반적인 list와 deque 모듈의 작업 시간을 비교하여 보다 효율적인 자료구조가 deque임을 보여주는데 똑같은 'append'와 'pop'을 사용하는데 왜 시간복잡도에서 차이가 발생할까?

정확한 이유에 대해서는 각 method가 데이터를 처리하는 방식에 대한 코드를 읽어보아야 하나 찾기 어려웠다.. 두 자료구조의 메소드 접근 방식에 따라 시간복잡도가 다를 것으로 예상했으나 시간복잡도는 같았다.

  - OrderDict
    - dictionary와 달리, 데이터를 입력한 순서대로 dict를 반환함(python 3.6 이전)
    - 현재는 dictionary 또한 순서가 기억되기 때문에 굳이 사용할 필요가 없음

  - defaultdict
   
    - default 값을 정해줌으로써, 신규 key 생성 시, value를 지정하지 않아도 default가 할당됨
    - dictionary에서는 key-value를 매칭함으로써 dictionary가 생성되지만 defaultdict에서는 key만 호출해도 default value가 매칭됨

  - Counter

    - count 메소드와 동일한 결과가 dictionary 형태로 반환됨
    - key는 element, value는 count결과로 매칭됨

  - namedtuple

    - tuple형태로 data 구조체를 저장하는 방법
    - 

❗️ 패킹과 언패킹

### 2. Pythonic code