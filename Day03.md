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

#### 3) tuple, 튜블

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


### 2. Pythonic code