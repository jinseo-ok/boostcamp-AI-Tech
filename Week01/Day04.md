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

##### (4) ㄷEncapsulation,  캡슐화



### 2. 