# Week01 - Python

## [Day 03] - 파이썬 기초 문법 II

### 1. Python Data Structure

이번 강의에서는 python에 존재하는 자료구조에 대해 배웁니다. 자료구조란 데이터를 저장할 때, 데이터에 특징에 따라 효율적으로 저장하기 위한 데이터의 저장 및 표현 방식을 의미한다. 자료구조에 따라 데이터를 저장하는 방식과 표현 및 특징이 다르기 때문에 상황에 따라서 적합한 자료구조로 효율적인 개발이 가능하다.

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

```python
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

```python
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

#### 6) collections

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

```python
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

---------

### 2. Pythonic code

이번 강의에서는 python의 특유 문법과 특징을 의미하는 **pythonic code**에 대해 배운다.
pythonic code는 지금까지 배운 자료구조 혹은 함수와 같이 정의된 정보 혹은 약속된 문법을 의미하는 것이 아닌 python 문법과 특징을 최대한 활용해서 가독성이 높은 깔끔한 코드를 작성하는 것을 의미한다.

pythonic code와 관련된 재밌는 [유튜브 영상](https://www.youtube.com/watch?v=Txz7K6Zc-_M)도 있다.
pythonic code와 주로 관련된 python 문법과 특징으로 대표적인 것들은 다음과 같다.

  - [x] **split & join**
  - [x] **list comprehension**
  - [x] **enumerate & zip**
  - [x] **lambda & map & reduce**
  - [x] **generator**
  - [x] **asterisk**


#### 1) split & join

split과 join에 method의 경우에는, 본래 알고있던 내용이긴 했다. 하지만 공식적으로 전달하고자 하는 내용에 대해서 한번 살펴보았다.
개인적으로는 내부 코드를 확인하고 싶었는데 어떻게 하는지 모르겠다..

[reference](https://docs.python.org/ko/3/library/stdtypes.html?highlight=split#str.join)

  - **str.split(sep=None, maxsplit=-1)**
>> sep 를 구분자 문자열로 사용하여 문자열에 있는 단어들의 리스트를 돌려줍니다.
>> maxsplit 이 주어지면 최대 maxsplit 번의 분할이 수행됩니다 (따라서, 리스트는 최대 maxsplit+1 개의 요소를 가지게 됩니다). maxsplit 이 지정되지 않았거나 -1 이라면 분할 수에 제한이 없습니다 (가능한 모든 분할이 만들어집니다).
>> sep 이 주어지면, 연속된 구분자는 묶이지 않고 빈 문자열을 구분하는 것으로 간주합니다 (예를 들어, '1,,2'.split(',') 는 ['1', '', '2'] 를 돌려줍니다).
>>sep 인자는 여러 문자로 구성될 수 있습니다 (예를 들어, '1<>2<>3'.split('<>') 는 ['1', '2', '3'] 를 돌려줍니다). 지정된 구분자로 빈 문자열을 나누면 [''] 를 돌려줍니다.
>> sep 이 지정되지 않거나 None 이면, 다른 분할 알고리즘이 적용됩니다: 연속된 공백 문자는 단일한 구분자로 간주하고, 문자열이 선행이나 후행 공백을 포함해도 결과는 시작과 끝에 빈 문자열을 포함하지 않습니다.
>> 결과적으로, 빈 문자열이나 공백만으로 구성된 문자열을 None 구분자로 나누면 [] 를 돌려줍니다.

```python
'1,2,3'.split(',')
-> ['1', '2', '3']
'1,2,3'.split(',', maxsplit=1)
-> ['1', '2,3']
'1,2,,3,'.split(',')
-> ['1', '2', '', '3', '']
```

  - **str.join(iterable)**
>>iterable 의 문자열들을 이어 붙인 문자열을 돌려줍니다.
>>iterable 에 bytes 객체나 기타 문자열이 아닌 값이 있으면 TypeError 를 일으킵니다. 요소들 사이의 구분자는 이 메서드를 제공하는 문자열입니다.

```python
list1 = ['1','2','3','4']  
s = "-"
s.join(list1) 
-> 1-2-3-4

list1 = ['A','I','c','a','m','p']
"_".join(list1)) 
-> A_I_c_a_m_p 
```

#### 2) list comprehension

  - 기존 List를 사용하여 간단히 다른 List를 만드는 기법
  - 포괄적인 List, 포함되는 List라는 의미로 사용됨
  - python에서 가장 많이 사용되는 기법 중 하나임
  - 일반적으로 for + append 보다 속도가 빠름 (위 유튜브 영상에도 있음!)

```python
list1 = 'ABCDEF'
list2 = 'HGDVEA'

result = []
for i in list1:
  for j in list2:
    if i == j:
      result.append(i+j)

result = [i+j for i in list1 for j in list2 if i == j] # else가 없을 때,

result = [i+j if i == j else ? for i in list1 for j in list2 ]  # else가 있을 때,
```

#### 3) enumerate & zip

  - **enumerate(iterable, start=0)**: list의 element를 추출할 때, index를 붙여서 추출 [ref](https://docs.python.org/ko/3/library/functions.html#enumerate)

>>열거 객체를 돌려줍니다.
>>iterable 은 시퀀스, 이터레이터 또는 이터레이션을 지원하는 다른 객체여야 합니다.
>>enumerate() 에 의해 반환된 이터레이터의 __next__() 메서드는 카운트 (기본값 0을 갖는 start 부터)와 iterable 을 이터레이션 해서 얻어지는 값을 포함하는 튜플을 돌려줍니다.

```python
def enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1
```

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
list(enumerate(seasons, start=1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

  - **zip(*iterables)**: 각 iterables(ex. list, array...) 의 요소(element)들을 모으는 이터레이터를 만듦 [ref](https://docs.python.org/ko/3/library/functions.html#zip)

>>튜플의 이터레이터를 돌려주는데, i 번째 튜플은 각 인자로 전달된 시퀀스나 이터러블의 i 번째 요소를 포함합니다.
>>이터레이터는 가장 짧은 입력 이터러블이 모두 소모되면 멈춥니다.
>>하나의 이터러블 인자를 사용하면, 1-튜플의 이터레이터를 돌려줍니다. 인자가 없으면, 빈 이터레이터를 돌려줍니다.

```python
def zip(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)
```

#### 4) lambda & map & reduce

  - **lambda**
      - 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수
      - 함수를 작성하지 않고 빠르게 사용함으로써 신속한 코드 작성 가능
      - 작동 여부 테스트의 어려움
      - 문서화 docstring 지원 미비
      - 코드 해석의 어려움

  - **map(function, iterable, ...)** [ref](https://docs.python.org/ko/3/library/functions.html#map)

>>iterable 의 모든 항목에 function 을 적용한 후 그 결과를 돌려주는 이터레이터를 돌려줍니다.
>>추가 iterable 인자가 전달되면, function 은 그 수 만큼의 인자를 받아들여야 하고 모든 이터러블에서 병렬로 제공되는 항목들에 적용됩니다.
>>다중 이터러블의 경우, 이터레이터는 가장 짧은 이터러블이 모두 소모되면 멈춥니다.

  - **filter(function, iterable** [ref](https://docs.python.org/ko/3/library/functions.html#filter)

>>function 이 참을 돌려주는 iterable 의 요소들로 이터레이터를 구축합니다.
>>iterable 은 시퀀스, 이터레이션을 지원하는 컨테이너 또는 이터레이터 일 수 있습니다.
>>function 이 None 이면, 항등함수가 가정됩니다, 즉, 거짓인 iterable 의 모든 요소가 제거됩니다.

built-in function인 **map**과 **filter**는 내가 굉장히 많이 사용하는 function이다. 가끔 사용하면서 헷갈릴 때가 있었는데 공식문서를 보면서 차이점을 분명히 알게 되었다.
map의 경우에는 모든 요소에 적용 후 반환까지, filter의 경우에는 모든 요소에 적용 후 False는 제거한다.

  - **functools.reduce(function, iterable[, initializer])** [ref](https://docs.python.org/ko/3/library/functools.html#functools.reduce)

>>두 인자의 function을 왼쪽에서 오른쪽으로 iterable의 항목에 누적적으로 적용해서, 이터러블을 단일 값으로 줄입니다.
>>예를 들어, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])는 ((((1+2)+3)+4)+5)를 계산합니다.
>>왼쪽 인자 x는 누적값이고 오른쪽 인자 y는 iterable에서 온 갱신 값입니다.
>>선택적 initializer가 있으면, 계산에서 이터러블의 항목 앞에 배치되고, 이터러블이 비어있을 때 기본값의 역할을 합니다.
>>initializer가 제공되지 않고 iterable에 하나의 항목만 포함되면, 첫 번째 항목이 반환됩니다.

```python
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
```

#### 5) iterable objects

  - Sequence형 자료형에서 데이터를 순서대로 추출하는 object
  - 내부적으로 `__next__`, `__iter__`가 구현되어 있음
  - iter(), next() 함수로 iterable 객체를 iterator object로 사용

```python
cities = ['Seoul', 'Busan', 'Jeju']
iter(cities)
-> <list_iterator at 0x160ace723d0> # 데이터가 담긴 메모리 주소를 의미함
memory_addr = iter(cities)

next(memory_addr)
-> 'Seoul'
next(memory_address)
-> 'Busan'
```

#### 6) generator

  - iterable object를 특수한 형태로 사용해주는 함수
  - element가 사용되는 시점에 값을 메모리에 반환
  - yield를 사용해 한번에 하나의 element만 반환
  - 큰 데이터를 처리할 때는 generator expression을 고려
  - 파일 데이터를 처리할 때도 generator를 사용하면 효율적

```python
def generator(value):
  result = []
  for i in range(value):
    yield i

generator(50)
-> <generator object generator at 0x00000160AEB61510> # generator 형태

for i in generator(50):
  print(i)

-> 0, 1, 2, 3
```
메모리에는 값의 주소만을 가지고 있고 호출할 때만 주소를 통해 값을 반환하는 것을 generator 형태로 이해했다.
이렇게 값이 아닌 메모리 주소를 저장하고 있음으로써 메모리 용량을 매우 효율적으로 관리할 수 있다.

  - generator compregension
    - generator expression이라고도 불림
    - '[ ]' 대신 '( )'를 사용함

```python
gen_ex = (n*n for n in range(500))
type(gen_ex)
-> <class 'generator>
```

#### 7) function passing arguments

함수에 입력되는 arguments의 다양한 형태

  - Keyword arguments
    - 인자의 parameter명이 정해져 있음
    - 인자를 입력할 때, parameter명을 매칭해주면 순서를 고려하지 않아도 됨
    - 하지만 인자만을 넣게되면 순서대로 parameter와 매칭됨
```python
def function(a, b, c):
  print(a, b, c)
  ....

function(c = 'top', a = 'middle', b = 'bottom')
-> 'middle' 'bottom' 'top'

function('top', 'middle', 'bottom')
-> 'top', 'middle', 'bottom'
```
  - Default arguments
    - paramter의 기본값을 지정함으로써, 입력되지 않을 경우 기본값 출력
    - 기본값이 지정되는 parameter는 좌측 정렬되어 있어야함( `def (a = '', b, c = '')` X) 
```python
def function(a, b, c = 'default'):
  print(a, b, c)
  ....

function('top', 'middle')
-> 'top', 'middle', 'default'

function('top', 'middle', 'bottom')
-> 'top', 'middle', 'bottom'
```

#### 8) Variable-length asterisk

함수의 parameter의 개수가 항상 정해져 있는 것은 아니다.
상황에 따라 적은 혹은 많은 parameter를 받고 함수가 작동되기를 원할 때에 **가변인자**를 통해 parameter를 사용할 수 있다.

##### (1) 가변인자

  - 개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 방법
  - keyword arguments와 함께, argument 추가가 가능
  - Asterisk(*) 기호를 사용하여 함수의 parameter를 표시함
  - 입력된 값은 tuple type으로 사용할 수 잇음
  - 가변인자는 오직 한 개만 맨 마지막(최좌측) parameter 위치에 사용 가능

```python
def function(a, b, *args):
  return a+b+sum(args)

function(1,2, 3,4,5)
-> 15

function(1,2, [3,4,5])
-> TypeError: unsupported operand type(s) for +: 'int' and 'list'

function(1,2, *[3,4,5])
-> 15
```

##### (2) 키워드 가변인자

```python
def function(a, b=2, *args, **kwargs):
  print(a)
  print(b)
  print(args)
  print(kwargs)

function(3,4,5,6,7,first = 1, second = 2, third = 3)
-> 3
-> 4
-> (5,6,7)
-> {'first' : 1, 'second' : 2, 'third' : 3}

function(3,first = 1, second = 2, third = 3)
-> 3
-> 2
-> ()
-> {'first' : 1, 'second' : 2, 'third' : 3}


function(a = 3, 4, 5, 6, first = 1, second = 2, third = 3)
 # a = 3으로 keyword를 지정해줬으면 그 뒤에도 계속 지정해줘야함
-> SyntaxError: positional argument follows keyword argument
function(a = 3, b = 4, args = (5, 6), first = 1, second = 2, third = 3)
-> 3
-> 4
-> ()
-> {'args': (5, 6), 'first': 1, 'second': 2, 'third': 3}
 # 그런데 args가 args로 묶이지 않고 키워드 가변인자로 되버림
```

##### (3) asterisk(*)

  - 곱셉, 제곱 연산, 가변 인자 등에서 다양하게 사용됨
  - tuple, dict 등 자료형에 들어가 있는 값을 unpacking
  - 함수의 입력값, zip 등에 유용하게 사용됨

```python
def function(a, *args):
  print(a, args)

function(1, *(2,3,4,5))
-> 1 (2, 3, 4, 5)

function(1, (2,3,4,5))
-> 1 ((2, 3, 4, 5),)

def function(a, args):
    print(a, args)

function(1, (2,3,4,5))
-> 1 (2, 3, 4, 5)

function(1,2,3,4,5)
-> TypeError: function() takes 2 positional arguments but 5 were given
```
