# Week01 - Python

## [Day 04] - 파이썬 기초 문법 III

### 1. Python Object Oriented Programming

이번 강의에서는 객체 지향 프로그래밍 언어, Object Oriented Programming(OOP)에 대해서 배운다.
OOP는 프로그래밍 언어를 배우는 데 있어서 매우 중요한 개념입니다. python도 OOP 형태로 구성되어 있기도 하지만, python 이외의 언어인 JAVA, C++, C# 같은 언어들이 모두 OOP 기반의 언어이다.
OOP를 배우는 것은 이전에 우리가 if 문이나 loop문을 배우듯이 프로그래밍 언어를 배우는 데 있어 가장 기본적인 개념이다.


#### 1) 개요

  - 객체(Object): 실생활에서 일종의 물건
    - **속성(Attribute)** 과 **행동(Action)** 을 가짐
  - OOP는 이러한 객체 개념을 프로그램으로 표현
  - 속성은 **변수(variable)**, 행동은 **함수(method)** 로 표현됨
  - OOP는 설계도에 해당하는 **클래스(class)** 와 실제 구현체인 **인스턴스(instance)** 로 구분됨

#### 2) Objects in Python

##### (1) class in Python

```python
class SoccerPlayer(object):
    def __init__(self, name, position, back_number):
        self.name = name
        self.position = position
        self.back_number = back_number

    def change_backNumber(self, new_number):
        print('선수의 등번호를 변경합니다: From {} to {}'.format(self.back_number, new_number))
        self.back_number = ner_number
```

  - class명 작성: class명을 작성할 때는, 보통 CamelCase 사용함
  - 속성(Attribute) 추가: `__init__`(객체 초기화 예약 함수),  `self`를 사용
  - 함수(method) 구현: 반드시 `self`를 추가해야만 class 함수로 인정됨

❗️`self`는 생성된 instance 자신을 의미함
❗️python에서 `__`는 특수한 예약 함수나 변수 그리고 함수명 변경(맨글링)으로 사용

#### 3) 특징

  - Inheritance, 상속성
  - Polymorphism, 다형성
  - Visibility, 가시성

##### (1) Inheritance, 상속성

부모 class를 상속함으로써, 부모 class의 속성과 행동을 모두 그대로 사용할 수 있는 것을 의미한다. 하지만 class 자체는 다른 class로 정의된다.

```python
class Person(object): # 부모 class
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Korean(Person): # 자식 class
    def __init(self, name, age, language):
        super().__init(name, age) # super()는 부모 class의 속성을 그대로 가져오는 것을 의미함
        self.language = language # language는 부모 class에 존재하지 않았던 속성
```

##### (2) Polymorphism, 다형성

부모 class에서 상속받은 같은 이름의 method를 overriding하여 기능을 확장하거나 변경하는 것을 의미한다. 
❗️overriding: 슈퍼클래스를 상속받은 서브클래스에서 슈퍼클래스의 메소드를 같은 이름, 같은 반환값, 같은 인자로 메소드 내의 로직을 새롭게 정의하는 것

##### (3) Visibility, 가시성

객체의 정보를 볼 수 있는 권한을 조절하는 것을 의미한다. 

  - 객체를 사용하는 사용자가 임의로 정보 수정하는 것을 막기 위함
  - 필요 없는 정보에 대한 접근 금지

##### (4) Encapsulation,  캡슐화

하나의 객체에 대해 그 객체가 특정한 목적을 위한 필요한 변수나 메소드를 하나로 묶는 것을 의미한다.

  - class를 설계할 때, 객체가 특정 목적을 잘 수행할 수 있도록 사용해야할 변수와 액션인 method를 잘 구성해야함

```python
# 일반적인 class 구현
class Inventory(object):
    def __init__(self):
        self.items = [] 
    def add_newItem():
        pass

my_inventory = Inventory()
my_inventory.items
-> []

# items 가시성 조절
class Inventory(object):
    def __init__(self):
        self.__items = [] # __ 로 items를 접근하지 못하도록 함
    def add_newItem()

my_inventory = Inventory()
my_inventory.__items
-> AttributeError: 'Inventory' object has no attribute 'items'

# poperty decorator
class Inventory(object):
    def __init__(self):
        self.__items = [] # __ 로 items를 접근하지 못하도록 함
    
    @property # property decorator, 숨겨진 변수를 반환하게 해줌
    def items(self):
        return self.__items

my_inventory = Inventory()
my_inventory.items
-> []
```

#### 4) decorator

##### (1) first-class objects

  - 일등 함수 또는 일급 객체(python의 함수는 모두 일급 함수)
  - 변수나 데이터 구조에 할당이 가능한 객체
  - 함수를 파라메터로 전달이 가능 + 리턴 값으로 사용(보다 유연한 함수 작성 가능)

```python
def f(x):
    return x * x
area = f
area(5)
-> 25
```

##### (2) Inner function

  - 함수 내에 또 다른 함수가 존재
  - closures: inner function을 return값으로 반환

##### (3) decorator function

  - 복잡한 closure 함수를 간단하게 만들어줌

---------

### 2. Module and Project

이번 강의에서는 파이썬 프로젝트의 기본이 되는 모듈과 패키지, 그리고 프로젝트의 개념에 대해서 배운다.
우리는 이미 파이썬에서 제공하는 다양한 모듈들을 사용했다. 이러한 모듈과 패키지를 구성하고 실제로 다른 개발자가 만든 모듈을 사용하는 방법에 대해서 다루게 된다.

#### 1) Module, 모듈

python으로 프로그램을 개발하기 위해 코드를 작성하게 되면 작게는 수천, 많게는 수만~수억 라인의 코드가 필요하다. 이렇게 프로그램이 길어짐에 따라서, 유지 및 보수를 보다 쉽게 하기 위해서 여러 개의 파일로 구분하는 방법이 있다.

이 때, 각 프로그램의 역할에 따라서 코드 및 프로그램을 모듈로 나눠서 관리할 수 있다. 즉, 모듈은 파이썬 정의와 문장들을 담고 있는 파일이며 모듈이 모여 더 큰 프로그램을 이루게 된다.



#### 2) package

