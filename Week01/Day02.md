# Week01 - Python

## [Day 02] - 파이썬 기초 문법

### 1. Variables

이번 강의에서는 프로그래밍에서 가장 기초적으로 알아야 할 변수에 대해서 배웁니다. 
변수와 함께 그 변수가 물리적으로 저장되는 메모리에 대해서 배웁니다.
마지막으로, 한 번에 여러 개의 변수를 함께 저장해서 화면에 표시하거나 필요한 연산을 수행하게 하는 리스트 라는 데이터 타입에 대해서 배웁니다.


#### 1) 변수, variable

  - 가장 기초적인 프로그래밍 문법 개념
  - 값(데이터, value)을 저장하기 위한 메모리 공간의 프로그래밍상 이름 -> 값(데이터, value)을 변수라는 공간에 **할당**하여 저장하는 개념
  - 정확히 변수라는 공간은 메모리 주소를 가지고 있고 변수에 값이 선언되는 순간, 메모리 특정 영역(메모리 주소)에 물리적인 공간이 할당됨
  
##### (1) 변수 명명 규칙

  - 알파벳, 숫자, 언더스코어(_)로 선언 가능
  - 변수명은 의미 있는 단어로 표기하는 것이 좋음 (코드 가독성을 위해 문맥과 역할을 활용할 필요가 있다고 생각함)
  - 변수명은 대소문자가 구분됨
  - 특별한 의미가 있는 예약어는 쓰지 않음 (예약어로 선언하게 되면 SyntaxError: invalid syntax 발생)
  - 데이터 타입으로 변수 선언이 가능하지만 권장하지 않음
  ```
  list = 'a'
  print(list) -> 'a'
  list('abc') -> TypeError: 'str' object is not callable
  ```
  ❗️ 예약어: 특정 기능을 수행하도록 미리 예약되어 있는 단어를 말한다. 파이썬 3.7 기준으로 33개의 예약어가 있다. [ref.](https://humahumahuma.tistory.com/2)

##### (2) Basic Operation (간단한 연산)

  - 기본 자료형 (primitive data types) : **데이터 타입에 따라 메모리 공간을 차지하는 용량이 달라짐**
  
  ![image](https://user-images.githubusercontent.com/48677363/104979234-1cf61800-5a47-11eb-8c8b-2b945acbb0e4.png)


##### (3) Dynamic Typing: 코드 실행시점에 데이터의 type을 결정하는 방법

  - python, scala와 같은 인터프리터 언어에서는 할당하고자 하는 값의 type에 따라 변수의 type을 정의하지 않아도 됨
  - JAVA와 같은 컴파일러 언어에서는 할당하고자 하는 값의 type에 따라 변수의 type을 정의해줘야 함

##### (4) 연산자(Operator)와 피연산자(operand)

  - +,-,*,/ 같은 기호들을 연산자라고 칭함
  - 연산자에 의해 계산이 되는 숫자들은 피연산자라 칭함
  - `3 + 2` 에서 '3'과 '2'는 피연산자, '+'는 연산자임
  - 수식에서 연산자의 역할은 수학에서 연산자와 동일함
  - 연산의 순서는 수학에서 연산 순서와 같음
  - 문자간에도 + 연산이 가능함, concatenate ('abc' + 'def' -> 'abcdef')
 
##### (5) 데이터 타입 변환

개인적으로 데이터 핸들링을 하게 되면 데이터 타입 변환에서 실수 및 에러가 굉장히 많이 발생하게 된다. 보통 다루게 되는 데이터 타입으로는 int(정수형), float(실수형), str(문자형)이 존재한다.

  - python은 데이터 타입을 유연하게 파악하여 변환이 가능하게 해주지만 float 형태의 str은 int로 변환 불가능하다.
  
  ```
  a = '3.0'
  int(a)
  -> ValueError: invalid literal for int() with base 10: '3.0'
  ```
  
  - float('nan') 혹은 None은 str과 float으로 변환이 가능하지만 int로 변환 불가능하다.
  
  ```
  pd.Series(list(np.arange(1, 50, 0.1)) + [float('nan')]).astype(int)
  -> ValueError: Cannot convert non-finite values (NA or inf) to integer
  ```
  
  ❗️ 여기서 새롭게 알게 된 사실은, None != float('nan')이라는 것은 알고 있었지만, None이 값으로 변환되면 float('nan')으로 변경되어 최종적으로는 두 값의 비교가 True로 반환된다.

### 2. Function and Console I/O

이번 강의에서는 콘솔 인/아웃에 대해서 배웁니다.
함수란 프로그램을 개발할 때 사용되는 코드의 논리적 단위로, 가장 기본적인 코드 개발 방법 중 하나입니다.
또한, 콘솔은 터미널이라고 불리는 컴퓨터 프로그램으로 컴퓨터에게 뭔가를 입력해서 컴퓨터가 결과를 출력하게 할 때 사용합니다.


#### 1) Function, 함수

어떤 일을 수행하는 코드의 덩어리.

  - 반복적인 수행을 1회만 작성 후 호출
  - 코드를 논리적인 단위로 분리
  - 캡슐화: 인터페이스만 알면 타인의 코드 재사용 가능

##### (1) 함수 선언 문법

  - 함수 이름: 함수와 함수를 구분하기 위한 명칭
  - parameter: 함수의 입력 값 인터페이스
  - argument: 실제 parameter에 대입되는 값
  - indentation: 코드블럭을 구분하기 위한 장치
  - return value(optional): 함수를 실행했을 때 반환되는 값

##### (2) 함수 수행 순서

  - 함수 선언 시, 함수를 메모리에 할당함 -> 코드 상단에 작성하는 것이 일반적임
  - 메인 프로그램 실행 시, 함수가 발견되면 메모리에 저장되어 있던 함수를 호출하여 수행됨

##### (3) 함수 형태

|            |            parameter 없음          |               parameter 존재             |
|:----------:|:---------------------------------:|:---------------------------------------:|
|  반환 값 없음 |         함수 내의 수행문만 수행         |        parameter를 사용, 수행문만 수행       |
|  반환 값 존재 | parameter없이, 수행문 수행 후 결과값 반환 | parameter를 사용하여 수행문 수행 후 결과값 반환 |


#### 2) Console I/O



##### (2) formatting

  - % string
  
```
print('Art: %5d, Price per Unit: %8.2f' % (453, 59.058))

-> Art: 453(5칸의 공간), Price per Unit: 59.06(8칸의 공간에 소수점 2자리까지..)
```

| type |  Description  |   
|:----:|:-------------:|
|  %s  | 문자열(string)   |
|  %c  |   문자 1개(character)    |  
|  %d  |   정수(Integer)    |  
|  %f  |   부동소수(floating-point)    |  
|  %o  |   8진수    | 
|  %x  |   16진수    | 
|  %%  |   Literal % (문자 % 자체)    | 

  
  
  - format 함수
  
```
age = 27
name = 'Jinseok'
print('{} is {} years old'. format(age, name))

-> 'Jinseok is 27 years old.'
```

```
print('Product: %10s, Price per unit: %10.3f.' % ('Apple', 5.243))
print('Product: {0:<10s}, Price per unit: {1:<10.3f}.'.format('Apple, 5.243')

-> Product:      APPLE, Price per unit:      5.243.
-> Product: Apple     , Price per unit: 5.243     .
```
  
  - fstring


### 3. Conditionals and Loops

이번 강의에서는 프로그래밍을 배울 때 사용되는 논리적인 사고 학습에 핵심이 되는 조건문과 반복문을 배웁니다. 거의 모든 프로그램은 제어와 반복의 연속으로 하나의 프로그램을 구성하게 됩니다. 사용자가 어떤 입력을 했을 때 그 입력에 따라 다양한 반응을 보이게 프로그램을 작성해야 하는데, 그를 이해하기 위해 필수적으로 배워야 하는 내용입니다. 사실 프로그래밍을 굳이 모르더라도 Computational Thinking 이라고 불리는 프로그래밍적 사고력을 위해서는 이해를 해야 할 부분입니다.
또한, 프로그램을 작성하면서 여러가지 실수를 하는데 그 실수를 수정하는 방법인 debug 의 개념과 방법을 공부하게 됩니다.


#### 1) 조건문

  - 조건에 따라 특정한 동작을 하게하는 명령문
  - 조건을 나타내는 기준과 실행해야 할 명령으로 구성됨
  - python에서는 조건문을 if, else, elif 등의 예약어를 사용하여 표현함

##### (1) 비교연산자, == 과 is

  - is 연산: 메모리의 주소를 비교
  - == 연산: 값을 비교
```
a = [1,2,3,4,5]
b = a[:] # copy
a is b
-> False
a == b
-> True
```
```
a = [1,2,3,4,5]
b = a # assign
a is b
-> True
a == b
-> True
```

##### (2) 삼항 연산자(Ternary operators): 조건문의 참과 거짓의 결과를 한줄에 표현하는 방식

```
values = True if 5 % 2 == 0 else False
```

#### 2) 반복문

  - 정해진 동작은 반복적으로 수행하게 하는 명령문
  - 반복문은 반복 시작 조건, 종료 조건, 수행 명령으로 구성됨
  - python에서는 for, while 등의 예약어를 사용하여 표현
  - 임시적인 반복 변수는 대부분 i, j, k로 할당
  

##### (1) for loop

  - 기본적인 반복문: 반복 범위를 지정하여 반복문 수행
  - 간격, 역순을 적용할 수 있음
  
##### (2) while

  - 조건이 만족하는 동안 반복 명령문을 수행
  - 반복 실행횟수가 명확하지 않을 때, 주로 사용
  - for문을 while문으로 변환 가능
  


### 4. String and advanced function conc

이번 강의에서는 그동안 우리가 변수로만 봤던 문자열 타입에 대해서 배웁니다.
우리가 사용해야 할 많은 데이터는 일반적으로 문자로 되어 있습니다. 흔히 숫자라고 생각하는 1, 0.1 같은 숫자들도, 컴퓨터는 문자로 인식하는 경우가 많습니다. 이번 강의에서 다룰 string 은 데이터를 다룰 때 가장 많이 접하게 되는 문자열 데이터를 다루는 방법에 대해서 설명합니다. 많지 않은 내용이지만 가장 많이 사용하는 타입 중에 하나이므로 충분히 공부하고 넘어가면 좋겠습니다.
또한, 함수의 조금 더 높은 난이도의 개념을 배우게 됩니다. 함수는 코드를 나누는 좋은 기준이기도 하지만 메모리의 사용이나 변수의 사용방법 등 다양한 개념들을 배워야 합니다.


#### 1) String, 문자열

  - **시퀀스 자료형**으로 문자열 데이터를 메모리에 저장
  - 영문자 한 글자는 1byte의 메모리 공간을 사용
  
##### (1) Indexing, 인덱싱

  - 문자열의 각 문자는 개별 주소(offset)를 가짐
  - 인덱싱이란 주소를 사용해 할당된 값을 가져오는 행위
  - List 인덱싱과 같은 형태
  

❗️ 데이터 타입과 메모리 공간

**1 Byte의 메모리 공간이란**

  - 컴퓨터는 2진수로 데이터를 저장
  - 이진수 한 자릿수는 1bit로 저장
  - 즉, 1bit는 0 또는 1
  - 1 byte = 8, bit = 2^8=256 까지 저장 가능
  
각 데이터 타입 별로 메모리 공간을 할당 받은 크기가 다르다.
  
![image](https://user-images.githubusercontent.com/48677363/105041898-e8b24400-5aa6-11eb-958f-0576031a1061.png)

  - 메모리 공간에 따라 표현할 수 있는 숫자범위가 다름
  - 데이터 타입은 **메모리의 효율적 활용** 을 위해 매우 종요
  
  
#### 2) function - call by object reference

함수에서 parameter를 전달하는 3가지 방식

  - 값에 의한 호출, Call by Value: 함수의 parameter로 값을 전달
  - 참조에 의한 호출, Call by Reference: 함수의 parameter로 메모리 주소를 전달
  - 객체 참조에 의한 호출, Call by Object Reference
  
##### (1) 객체 참조에 의한 호출, Call by Object Reference

python에서는 **객체의 주소가 함수** 로 전달되는 방식이 적용된다. 전달된 객체를 참조하여 변경 시, 호출자에게 영향을 주나 새로운 객체를 만들 경우 호출자에게 영향을 주지 않는다. 

```
def spam(eggs):
    eggs.append(1) # paramter로 입력 받은 객체와 같은 메모리 주소
    eggs = [2, 3] # 새롭게 선언된 변수로 새로운 메모리 주소를 가지게 됨
    ham = [4, 5] # 그렇다면 여기서 ham을 새롭게 선언하면 어떻게 될까?, 하지만 해당 ham은 parameter로 들어오는 ham과는 전혀 다른 ham!
    print(eggs) # 새롭게 선언된 [2, 3] print
    print(ham) # 위에서 선언된 [4, 5] print

ham = [0]
spam(ham)
print(ham)
-> [0, 1] # 원소가 추가된 모습, ham -> ham(eggs).append(1) -> ham

```

**swap**

  - 함수를 통해 변수 간의 값을 교환(swap)하는 함수
  - Call By XXXX를 설명하기 위한 전통적인 함수 예시
  
```
def swap_value(x, y):
    temp = x
    x = y
    y = temp
    
def swap_offset(offset_x, offset_y):
    temp = ex[offset_x] 
    ex[offset_x] = ex[offset_y] 
    ex[offset_y] = temp

def swap_reference(list_ex, offset_x, offset_y):
    temp = list_ex[offset_x]
    list_ex[offset_x] = list_ex[offset_y]
    list_ex[offset_y] = temp

ex = [1,2,3,4,5]
swap_value(ex[0], ex[1])
-> ex = [1,2,3,4,5]

swap_offset(0, 1)
-> ex = [2,1,3,4,5]

ex = [1,2,3,4,5]
swap_reference(ex, 2, 3)
-> ex = [1,2,4,3,5]

```

#### 3) function - scoping rule

  - 변수가 사용되는 범위
  - 지역변수(local variable): 함수 내에서만 사용
  - 전역변수(Global variable): 프로그램 전체에서 사용
  
```
def f():
    global s # 함수 내에서 전역변수 사용 시, global 키워드 사용. 함수 내에서만 존재했던 s가 프로그램 전체 s로 변경됨
    s = 'I love London!'
    print(s)
    
s = 'I love Paris!'
f()
print(s)

-> I love London!
-> I love London!
```

#### 4) recursive Function, 재귀함수

  - 자기자신을 호출하는 함수
  - 점화식과 같은 재귀적 수학 모형을 표현할 때 사용
  - 재귀 종료 조건 존재, 종료 조건까지 함수호출 반복
  
#### 5) function type hints

  - python의 특징 중 하나인 dynamic typing은 익숙치 못한 사용자에게 interface가 어렵다는 단점이 있음
  - 함수의 parameter의 데이터 타입을 적음으로써, 보다 가독성 있게 해주는 type hints 기능을 제공함
    - 사용자에게 가독성 있는 인터페이스 제공
    - 함수 문서화 시, parameter 대한 정보 명시 가능
    - mypy 또는 IDE, linter 등을 통해 코드의 발생 가능한 오류를 사전에 확인 가능
    - 시스템의 전체적인 안정성 확보 가능
    
```
def insert(self, index: int, module: Module) -> None:
    .....
    .....
    
-> index는 int 타입, return은 None 임을 알 수 있음

```

#### 5) function docsting

  - python 함수에 대한 상세 스펙을 사전에 작성 -> 함수 사용자의 이행도를 확보 가능
  - 주석을 통해, docstring 영역 표시(보통 함수명 아래)
  
#### 6) 함수 개발 가이드 라인

  - [x] 함수는 가능하면 짧게 작성(코드 라인을 줄일 것)
  - [x] 역할, 의도가 명확하게 드러나도록 작성 (verb + object) + (underscore 사용)
  - [x] 하나의 함수에는 유사한 역할을 하는 코드만 포함
  - [x] 인자로 받은 값 자체를 바꾸지는 말 것(임시변수 선언)
  - [x] 공통 코드는 함수로 작성
  - [x] 복잡한 수식은 함수로 작성 -> 함수화로 복잡한 부분을 따로 설명 혹은 분리할 수 있음
