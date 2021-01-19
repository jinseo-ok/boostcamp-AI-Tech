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
  `
  list = 'a'
  print(list) -> 'a'
  list('abc') -> TypeError: 'str' object is not callable
  `
  * 예약어: 특정 기능을 수행하도록 미리 예약되어 있는 단어를 말한다. 파이썬 3.7 기준으로 33개의 예약어가 있다.[출처](https://humahumahuma.tistory.com/2)

##### (2) Basic Operation (간단한 연산)

  - 기본 자료형 (primitive data types) : **데이터 타입에 따라 메모리 공간을 차지하는 용량이 달라짐**
  
  ![image](https://user-images.githubusercontent.com/48677363/104979234-1cf61800-5a47-11eb-8c8b-2b945acbb0e4.png)


##### (3) Dynamic Typing: 코드 실행시점에 데이터의 type을 결정하는 방법


  


### 2. Function and Console I/O




### 3. Conditionals and Loops




### 4. String and advanced function conc




