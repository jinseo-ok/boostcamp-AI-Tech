# Week01 - Python

## [Day 05] - 파이썬으로 데이터 다루기

### 1. File / Exception / Log Handling

이번 강의에서는 프로그램을 보다 단단하고 치밀하게 만들어주는 예외 처리와 파일 다루기에 대해서 배운다.

개발 시, 다양한 상황에서 예상치 못한 오류가 발생합니다. 이를 해결하고 방지하기 위해서 사전에 오류가 발생할 수 있는 예외를 포괄적으로 지정해서 대비하게 됩니다.
또 프로그램을 개발할 때에는, 앞서 소개한 여러 모듈 및 파일과 폴더를 사용하는 경우가 많기 때문에 유동적으로 파일과 디렉토리를 다루는 것에 대해 다루게 됩니다.

추가적으로 프로그램을 진행하면서 진행 상황 및 이슈를 기록하는 로깅(logging)에 대해서 배운다. 로그 데이터를 남기는 것은 데이터 기반의 애플리케이션 개발에서 매우 중요하다.
로그 데이터를 보다 구조적으로 설계하는 것이 하나의 분야로 자리잡고 매우 중요한 부분으로 여겨지기 때문에 숙지하는 것이 좋다.

#### 1) Exception, 예외

##### (1) 예상이 가능한 예외

  - 발생 여부를 사전에 인지할 수 있는 예외
  - 사용자의 잘못된 입력, 파일 호출 시 파일 없음 등
  - 개발자가 반드시 명시적으로 정의(처리)해야함

##### (2) 예상이 불가능한 예외

  - 인터프리터 과정에서 발생하는 예외, 개발자 실수
  - 리스트 범위를 넘어가는 값 호출, 정수 0 으로 나눔 등
  - 수행 불가 시, 인터프리터가 자동 호출

##### (3) Exception Handling, 예외 처리

복잡한 프로그램을 개발할수록, 다양한 방식으로 예상할 수 없는 예외가 발생할 수밖에 없다. 당연히 모든 예외 사항을 예측하여 오류와 에러를 방지하는 것이 최선의 방법이지만, 현실적으로 불가능하다.

그렇기 때문에, 예외가 발생할 수 있는 부분 혹은 전체 시스템에서 예외가 발생할 수 있음을 가정하여, 그 부분에 예외 처리를 하고 기록함으로써 프로그램을 개선시키는 것이 중요하다.
그러므로 더욱 치밀하고 견고하게 예외 처리를 할수록 프로그램 동작에 이상이 없고 개선할 수 있는 여지도 충분히 갖출 수 있다.

  - try ~ except: 가장 대표적이며 기본적인 예외 처리 방법
```python
try:
    예외 발생 가능 코드
except <Exception Type> as e: # 어떤 내용의 exception이 발생했는지 볼 수 있음
    예외 발생 시, 대응 코드
```
  - built-in Exception: python 기본 예외 사항

| Exception |  Description  |   
|:----:|:-------------:|
|  IndexError  | List의 index가 존재하지 않을 때  |
|  NameError  |  존재하지 않은 변수를 호출 할 때  |  
|  ZeroDivisionError  |  0으로 숫자를 나눌 때   |  
|  ValueError  | 변환할 수 없는 문자/숫자를 변환할 때  |  
|  FileNotFoundError | 존재하지 않는 파일을 호출할 때  | 

```python
try:
    예외 발생 가능 코드
except:
    예외 발생 시, 대응 코드
else: # 그러나 if~else문은 로직 문제일 때 사용하는 것이 좋음
    예외가 발생하지 않을 때 실행 코드
finally:
    마지막에 무조건 실행하는 코드
```

  - raise 구문: 필요에 따라 강제로 Exception을 발생
```python
raise <Exception Type>(예외정보)

while True:
    value = input('숫자를 입력해주세요:')
    for digit in value:
        raise ValueError('숫자값이 아닙니다.')
```

  - assert 구문: 특정 조건에 만족하지 않을 경우 예외 발생, 보통 함수에서 중간중간 제대로 값이 잘 들어왔는지 확인하기 위해서 점검

```python
assert 예외조건

a = int
b = input()
assert a == type(b)
-> True or False
```

#### 2) File Handling

##### (1) 파일의 종류

  - 기본적인 파일 종류로 text 파일과 binary 파일로 나뉨
  - 컴퓨터는 text 파일을 처리하기 위해 binary 파일로 변환시킴, 즉 모든 text 파일은 binary 파일이라고 볼 수 있음


|  Binary 파일   |   Text 파일    |   
|:------------:|:-------------:|
|  컴퓨터만 해석할 수 있는 이진 형식으로 저장된 파일  | 시각적으로 해석 가능한 형식으로 저장된 파일 |
|  파일을 열면 깨지는 현상 발생  |  내용 확인 가능  |  
|  엑셀, 워드 파일  |  txt, html, py 파일 |

##### (2) 파일 읽기

  - python을 파일 처리를 위해 'open' built-in function을 사용함

|  접근 모드   |   Description   |   
|:------------:|:-------------:|
|  r  | 읽기모드 - 파일을 읽기만 할 때 사용(default) |
|  w  | 쓰기모드 - 파일에 내용을 쓸 때 사용  |  
|  a  | 추가모드 - 새로운 내용을 추가 시킬 때 사용 |

<br>

```sql
# introduction.txt 파일
Hi My name is Carvin.
I'm 27 years old.
I really want to be a recommendation system developer.
So, I have been studying Deep learning and Data engineering.
```

```python
with open('introduction.txt', 'r') as f:
  files = f.read()
-> b"Hi My name is Carvin.\nI'm 27 years old.\nI really want to be a recommendation system developer.\nSo, I have been studying Deep learning and Data engineering."
  
  files = f.readlines() # 파일 전체를 list로 반환
-> [b'Hi My name is Carvin.\n',
 b"I'm 27 years old.\n",
 b'I really want to be a recommendation system developer.\n',
 b'So, I have been studying Deep learning and Data engineering.']
  
  files = f.readline() # 파일 중 한줄씩 읽기
-> b'Hi My name is Carvin.\n'
```

##### (3) 파일 쓰기

```python
f = open('*.txt', 'w', encoding = 'utf8')
for i in range(1, 11):
  data = f'{i}번째 줄입니다.\n'
  f.write(data)
f.close()

with open('*.txt', 'a', encoding = 'utf8') as f: # mode = 'a'는 update 모드
  for i in range(1, 11):
    data = f'{i}번째 줄입니다.\n'
    f.write(data)
```

##### (4) 파일 다루기

  - os 모듈

os 모듈은 Operating System의 약자로서 운영체제에서 제공되는 여러 기능을 python에서 수행 가능하게 합니다.

|  method   |   Description   |   
|:------------:|:-------------:|
|  getcwd()  | 현재 작업 dir 확인 |
|  mkdir(name)  | 현재 dir에서 name 폴더 생성 |
|  listdir(dir)  | 해당 dir의 파일 조회(ls)  |  
|  path.exists(dir)   | 해당 dir의 파일 존재 여부 |
|  path.join('', '')   | 경로 만들어주기 |

<br>

  - pickle
    - python의 객체를 **영속화(persistence)** 하는 built-in 객체
    - 데이터, object 등 실행 중 정보를 binary로 저장 -> 불러와서 사용
    - 저장해야하는 정보, 계산 결과(모델) 등 활용을 많이 함

❗️persistence, 영속성이란 ‘영원히 계속되는 성질이나 능력’을 의미하며, 당시 상태를 그대로 저장한다는 것을 의미한다고 생각함

#### 3) Logging Handling

log, 로그란 시스템의 기록을 담고 있는 데이터를 의미한다. 이러한 데이터에는 성능, 오류, 경고 및 운영 정보 등의 중요 정보가 기록되며, 특별한 형태의 기준에 따라 숫자와 기호 등으로 이루어져 있다.

python에서는 log를 추적하고 기록하기 위해 logging 모듈을 제공하고 있다. logging은 어떤 소프트웨어가 실행될 때 발생하는 이벤트를 추적하는 수단이다. 소프트웨어 개발자는 코드에 logging 호출을 추가하여 특정 이벤트가 발생했음을 나타내고 확인할 수 있다.

  - 프로그램이 실행되는 동안 일어나는 정보를 기록으로 남기기
  - 유저의 접근, 프로그램의 Exception, 특정 함수의 사용 등이 기록이 될 수 있음
  - Console 화면, 파일, DB 등 다양한 위치에 기록 및 저장할 수 있음
  - 기록된 로그를 분석하여 의미있는 결과를 도출하고 프로그램을 개선할 수 있음
  - 실행시점에서 남겨야 하는 기록, 개발시점에서 남겨야하는 기록

**logging 모듈**

  - 프로그램 진행 상황에 따라 다른 Level의 Log를 출력함
  - 개발 시점, 운영 시점 마다 다른 Log가 남을 수 있도록 지원함
  - Log Level: DEBUG > INFO > WARNING > ERROR > CRITICAL
  - Log 관리 시, 가장 기본이 되는 설정 정보
  - python의 기본 logging level은 WARNING -> 변경 가능

![image](https://user-images.githubusercontent.com/48677363/105492864-f01e5b00-5cfb-11eb-901d-6ba3ae3ec859.png)

```python
import logging

if __name__ == '__main__':
  logger = logging.getLogger('main')
  logging.basicConfig(level = logging.DEBUG) # DEBUG로 logging level 하향 조정
  logger.setLevel(logging.INFO)

  # logging 출력 (my_log 파일로 쓰기)
  steam_handler = logging.FileHander(
      'my_log', mode = 'w', encoding = 'utf8')
  logger.addHandler(steam_handler)
```

##### (1) logging 모듈 설정

  - **configparser**
    - 프로그램의 실행 설정을 file에 저장함
    - Section, Key, Value 값의 형태로 설정된 설정 파일을 사용
    - 설정파일을 Dict Type으로 호출 후 사용

  - **argparser**
    - Console 창에서 프로그램 실행 시 setting 정보를 저장함
    - 거의 모든 Console 기반 python 프로그램에서 기본으로 제공
    - 특수 모듈도 많이 존재하지만, 일반적으로 argparse를 사용
    - Command-Line Option 이라고 부름

```python
import argparse

args = argparse.ArgumentParser(description='Argparse Tutorial')
args.add_argument(
              dest = '--name', # 해당 객체
              default = '', # 기본값
              type = int, # 인자 type
              required = True, # 필요성 여부
              help='an integer for printing repeatably' # help 시 설명
            )
```

### 2. Python data handling

이번 강의에서는 python을 통해 데이터 분석에서 가장 많이 다루는 csv, web, XML, JSON이라는 네 가지 데이터 형식에 대해서 배운다.

CSV 파일 포맷은 어떠한 프로그래밍 언어든 데이터를 다루는 분야에서 가장 기본이 되는 형식이다.
보통 엑셀의 텍스트 데이터 형식을 CSV로 이해할 수 있다. 데이터프레임으로 다루기 편하며 데이터 분석 및 python에서 매우 중요한 데이터 타입으로 파이썬으로 핸들링하는 것에 익숙해지는 것이 필요하다.

웹은 인터넷 공간을 의미하며 가장 많은 정보를 제공하고 있다. 그렇기 때문에 웹 상에 존재하는 데이터를 통해서 다양한 분석을 할 수 있게 된다. 웹 상에 존재하는 데이터를 얻기 위해서는 웹을 표현하는 가장 기본적인 언어인 HTML을 배우고 HTML을 가져올 수 있는 크롤링 기술을 배우게 된다.
특히, 웹 상 존재하는 raw한 데이터를 전처리하기 위해 정규표현식(regex)를 학습함으로써 데이터를 보다 빠르고 정확하게 처리할 수 있다.

마지막으로 데이터를 저장하는 다양한 포멧중 하나인 XML(eXtensible Markup Languages)과 JSON(JavaScript Object Notation)에 대해서 다루도록 한다.
XML 프로그래밍 언어에서 데이터를 저장하고 불러오는 전통적인 파일포맷은 흔히 레거시 시스템(오래전에 구축된 시스템)에서 raw파일을 저장하는 대표적인 포맷입이다. JSON은 이와 달리 모바일이 활발히 사용되면서 사용되기 시작하는 저장 포맷으로 웹에서 많이 사용되는 JavaScript의 문법을 활용하여 저장하는 포맷이다.

#### 1) CSV, Comma Seperate Value

  - CSV, 필드를 쉼표(,)로 구분한 텍스트 파일
  - 엑셀 양식의 데이터를 프로그램에 상관없이 쓰기 위한 데이터 형식(공유하기 쉬움)
  - 탭(TSV), 빈칸(SSV) 등으로 다양한 구분자를 통해 저장 가능
  - 통칭하여 character-separated values (CSV) 부름
  - 엑셀에서는 “다름 이름 저장” 기능으로 사용 가능
  - 구분자를 고려하여 전처리를 해야함(구분자가 ','일 때, 데이터값에도 ','이 존재할 가능성이 크기 때문)

```python
import csv
reader = csv.reader(
                  f,
                  delimiter = ',', # 구분자
                  quotechar = '"', # 문자열을 둘러쌓는 신호 문자
                  lineterminator = '', # 줄 구분자
                  quoting = csv.QUOTE_ALL # quotechar 레벨 지정
                )
```

#### 2) WEB

  - World Wide Web(WWW)
  - 팀 버너스리에 의해 1989년 처음 제안되었으며, 원래는 물리학자들간 정보 교환을 위해 사용됨
  - 데이터 송수신을 위한 HTTP 프로토콜 사용, 데이터를 표시하기 위해 HTML 형식을 사용

##### (1) WEB 구조

<img src = 'https://user-images.githubusercontent.com/48677363/105497063-b6e8e980-5d01-11eb-84e3-4dea1e4bbb1c.png' width = '500'>

##### (2) HTML, Hyper Text Markup Language

  - 웹 상의 정보를 구조적으로 표현하기 위한 방식
  - 제목, 단락, 링크 등 요소 표시를 위해 Tag를 사용
  - 모든 요소들은 꺾쇠 괄호 안에 둘러 쌓여있음

`<title> Hello, World </title> #제목 요소, 값은 Hello, World`

  - 모든 HTML은 트리 모양의 포함관계를 가짐
  - 일반적으로 웹 페이지의 HTML 소스파일은 컴퓨터가 다운로드 받은 후 웹 브라우저가 해석 후 표시

```html
<!doctype html>
<html>
  <head>
    <title>Hello HTML</title>
  </head>
  <body>
    <p>Hello World!</p>
  </body>
```

##### (3) 정규표현식

- 정규 표현식, regexp 또는 regex 등으로 불림
- 복잡한 문자열 패턴을 정의하는 문자 표현 공식
- 특정한 규칙을 가진 문자열의 집합을 추출
  - 주민등록 번호, 전화번호, 도서 ISBN 등 형식이 있는 문자열을 원본 문자열로부터 추출함
  - HTML역시 tag를 사용한 일정한 형식이 존재하여 정규식으로 추출이 용이함
- python에서는 정규표현식에 대한 모듈로 re 모듈을 제공함

|  method   |   Description   |   
|:------------:|:-------------:|
| match(pattern, string) | pattern으로 시작하는 string을 return |
| search(pattern, string) | pattern이 존재하는 string을 return |
| findall(pattern, string) | string에 존재하는 pattern을 return |  
| fullmatch(pattern, string | 정확하게 pattern과 일치하는 string을 return|
| sub(pattern, repl, string) | string에 존재하는 pattern을 repl로 변경 |

<br>

#### 3) XML, eXtensible Markup Language

  - 데이터의 구조와 의미를 설명하는 TAG(MarkUp)를 사용하여 표시하는 언어
  - TAG와 TAG사이에 값이 표시되고, 구조적인 정보를 표현할 수 있음
  - HTML과 문법이 비슷, 대표적인 데이터 저장 방식(초기 데이터 저장 및 전송 형식)
  - 정규표현식으로 parsing이 가능함

```xml
<?xml version="1.0"?>
<고양이>
  <이름>나비</이름>
  <품종>샴</품종>
  <나이>6</나이>
  <중성화>예</중성화>
  <발톱 제거>아니요</발톱 제거>
  <등록 번호>Izz138bod</등록 번호>
  <소유자>이강주</소유자>
</고양이>
```

#### 4) JSON, JavaScript Object Notation

  - json 모듈을 사용하여 손 쉽게 파싱 및 저장 가능
  - 원래 웹 언어인 Java Script의 데이터 객체 표현 방식
  - **간결성** 으로 기계/인간이 모두 이해하기 편함
  - 데이터 용량이 적고, Code로의 전환이 쉬움
  - 이로 인해 XML의 대체제로 많이 활용되고 있음
  - key-value의 표현으로 데이터 저장 및 읽기가 dict type과 상호 호환 가능
  - 웹에서 제공하는 API는 대부분 정보 교환 시 JSON 활용


 
 
