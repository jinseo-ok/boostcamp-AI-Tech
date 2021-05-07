# Week02 - AI Math

## [Day 08] - Pandas I / 딥러닝 학습방법 이해하기

### 1. Pandas I

panel data 의 줄임말인 **pandas**는 python의 데이터 처리의 가장 기본적이며 표준 라이브러리입니다.

pandas는 python에서 일종의 엑셀과 같은 역할을 하며, 데이터를 전처리하거나 통계 처리 시 많이 활용하는 피벗 테이블 등의 다양한 기능을 사용할 수 있습니다. pandas 역시 numpy를 기반으로 개발되어 있고 R의 데이터 처리 문법과 유사한 부분이 많기 때문에 생각보다 빠르게 익숙해질 수 있습니다.

  - 구조화된 데이터의 처리를 지원하는 python 라이브러리
  - 고성능 array 계산 라이브러이인 numpy와 통합하여 강력한 '스프레드시트' 처리 기능을 제공
  - 인덱싱, 슬라이싱, 연산용 함수, 전처리 함수 등을 제공함
  - 데이터 처리 및 통계 분석을 위해 사용함
  - tabular 데이터를 다루는 것에 가장 최적화 되어 있음

#### 1) DataFrame, 데이터프레임

데이터를 tabular 형태로 나타난 데이터 형식으로 엑셀의 스프레드시트와 유사한 형태입니다. 이 때, tabular 형태란 표 형식의 데이터를 의미하며 행과 열로 이루어진 형태라고 생각하면 될 것 같습니다. pandas에서 데이터를 다루게 되면, 데이터프레임, tabular data란 표현을 많이 사용하게 되는데, 표 형식의 데이터라고 이해해도 무방할 것으로 생각됩니다.

기본적으로 행과 열로 구성되어 있는 데이터프레임은 열(column)에는 속성(attribute)과 속성에 대한 값들이 담기게 되고 행(row)에는 행에 해당하는 속성 값들의 모음으로 데이터프레임은 행 데이터의 집합입니다. 

<image src = https://user-images.githubusercontent.com/48677363/105959646-83baa780-60bf-11eb-949b-a0e51d2650ac.png width = 600>

  - Numpy array-like
  - Series의 모음 object
  - 각 Series는 모두 다른 data type을 가질 수 있음
  - 각 row는 데이터프레임 고유의 index를 가지고 있으며, 모든 Series는 해당 index에 매핑되어 있음
  - 행렬과 마찬가지고 행과 열로 indexing이 가능함
    - **iloc**: index position search, 인덱스 위치로 검색
    - **loc**: index search, 인덱스로 검색

#### 2) Series, 시리즈

데이터프레임 중 하나의 columns에 해당하는 벡터를 표현하는 object입니다. Series는 벡터와 매핑되는 index와 데이터를 의미하는 벡터, 그리고 해당 벡터들의 data type으로 구성되어 있습니다.

<image src = https://user-images.githubusercontent.com/48677363/105960760-02fcab00-60c1-11eb-817f-d03e5d669d6c.png width = 300>

  - numpy.ndarray를 기반으로 한 subclass
  - iteration과 vectorize가 가능함
  - index로 데이터를 호출할 수 있음
  - dictionary 형태의 데이터를 Series로 변환할 수 있으며, key-value가 index-value로 변환됨

pandas에서는 Series 자료구조에게 분석에 유용한 메소드를 많이 제공합니다.

- `Series.value_counts()` -> Series의 벡터의 개수를 구해주는 method로 데이터가 categorical할 때 사용하는게 유용함
- `Series.sort_values()` -> Series의 벡터를 정렬하는 method로 default로는 오름차순으로 설정되어 있으며 continuous할 때 사용하는게 유용함
- `Series.nlargest()` -> sort_values(ascending = False)와 똑같은 결과를 보여주지만, 더 빠른 것으로 알고 있음
- `Series.to_dict()` -> {index : vector}, index를 key로 vector를 value로 가지는 dictionary 형태로 변환해주는 method
- `Series.astype(type)` -> 해당 Series의 type을 변환해주는 method, vector의 형태에 따라 변환이 불가능한 type이 존재할 수 있음

#### 3) data handling

##### (1) columns

  - columns을 선택할 때는 3가지 방법이 있음
    - `df[column]` -> pd.Series 형태로 column 출력 (여려 개의 columns 출력 불가)
    - `df[[column]]` -> pd.DataFrame 형태로 column 출력 (여려 개의 columns 출력 가능)
    - `df.iloc[:, 1:3]` -> iloc으로 슬라이싱하는 방법으로, 첫번째~두번째 column을 출력 가능
  - column 명을 변경하는 방법은 다양함
    - `df = df.rename(columns = {기존column명 : 변경column명})`
    - `df.columns = ['A', 'B', 'C']` -> 한번에 바꿀 때만 가능함, 슬라이싱해서 column명을 바꿀 수는 없음
  - column 삭제하는 방법
    - `del df[column]`
    - `df.drop(column, axis = 1)`

##### (2) Rows

  - row를 선택하는 방법 또한 다양함
    - `df.loc[index명]`
    - `df.iloc[index position]`
    - `df.loc[index명, column(s)]`
  - row의 index를 변경하는 방법
    - `df.index = 변경할 index` -> index의 개수가 일치해야함
    - `df.reset_index(drop = True)` -> 현재 position으로 index 초기화, drop = True로 기존 index 삭제
  - row를 삭제하는 방법
    - `df.drop(index, axis = 0)`

##### (3) 조건 추출

pandas의 데이터프레임은 행렬에서 행과 열을 조건에 따라 선택하는 것과 같이 조건에 따라 행과 열을 선별할 수 있습니다.

  - `df.loc[df[column] == 1(조건)]` -> 해당 column의 조건에 해당하는 행을 출력
  - `df.loc[(df[column] == 1) & (df[column] > 10)]`
  - `df.query('column == 1(조건)')` -> 해당 column의 조건에 해당하는 행을 출력
  - `df.query('column == @variable')` -> @ 표시로 변수를 인식할 수는 있음

##### (4) 데이터 연산

보통 데이터프레임에서 데이터 전처리를 위한 연산을 적용할 때에는 column을 기준으로 적용하는 경우가 많습니다. 보통 column과 column은 독립적이라고 가정하기 때문에 column에 따라 적용해야할 처리가 다르기 때문입니다.

  - `df.apply(f)` -> 모든 Series에 해당 f 함수를 적용하는 방법 
  - `df[column].apply(lambda x : f(x))` ->  해당 Series의 모든 벡터에 해당 f 함수를 적용하는 방법
  - `df[column].astype(int)` -> 해당 Series의 data type을 변경 방법
  - `df[column].diff()` -> 해당 Series의 앞과 뒤의 vector 차를 구하는 방법

##### (5) built-in Function

  - **describe**: Numeric type 데이터의 요약 정보를 반환
  - **unique**: Series data의 고유 값을 반환
  - **sum(axis = )**: 축을 기준으로 행과 열의 합산 값을 반환
  - **isnull**: 모든 데이터 값 == Null에 대한 결과를 반환
  - **fillna('')**: Null 값을 ''로 채워주는 결과를 반환
  - sort_values, value_counts....

----------

### 2. 딥러닝 학습방법 이해하기

이전 강의에서 배운 선형 모델은 단순하거나 저차원 데이터를 해석할 때는 유용하지만 분류 문제와 고차원 문제에 적용할 때에는 예측률이 높지 않습니다. 이를 개선하기 위해 비선형 모델인 신경망을 도입할 수 있고 보다 높은 예측률을 얻을 수 있습니다.

이번 강의에서는 신경망 구조에 존재하는 layer와 활성화 함수(activation function), 그리고 학습 과정인 역전파 알고리즘에 대한 기본적인 개념을 학습하게 됩니다.

#### 1) 신경망, Neural Network

Neural Network, 신경망은 뇌의 기본 구조 조직인 뉴런(neuron)과 뉴런이 연결되어 일을 처리하는 과정에서 영감을 얻은 통계학적 학습 알고리즘입니다.

화학적 또는 전지적 신호를 전송하고 처리하는 세포인 뉴런이 서로 연결되어 일을 처리하는 것처럼, 수학적 모델로서의 뉴런이 상호 연결되어 네트워크를 형성할 때 이를 **신경망, Neural Network**라고 합니다.

#### 2) 신경망 구조

신경망은 기본적으로 신호를 받아들이는 **입력층(input layer)**, 뉴런이 상호작용하는 **은닉층(hidden layer)**, 그리고 결과가 출력되는 **출력층(output layer)** 로 구성되어 있습니다.

신경망이라는 네트워크를 통해 데이터 간의 비선형적인 문제를 해결할 수 있는 이유는 뉴런이 상호작용하여 계산이 발생하는 은닉층에서 선형 계산과 비선형 계산이 모두 이뤄지기 때문입니다. 그러므로 은닉층의 층이 깊어질수록 선형 함수와 비선형 함수와의 조합을 바탕으로 어떠한 비선형적인 특징을 찾을 수 있는 것이 신경망의 가장 큰 장점입니다.

##### 소프트맥스

소프트맥스(softmax) 함수는 모델의 출력을 확률로 해석할 수 있게 변환해주는 연산을 가집니다. 분류 문제, classification에서 어떠한 class에 분류될 확률을 반환해줍니다.

<image src = https://user-images.githubusercontent.com/48677363/106079476-a4830b80-6158-11eb-9556-a0c8a04e656f.png width = 500>

<br>
<br>

```python
output = np.array([[1,2,3], # 마지막 layer의 출력값
                   [0,1,3],
                   [1,0,0]])

def softmax(x):
    denumerator = np.exp(x - np.max(x, axis = -1, keepdims = True))
    numerator = np.sum(denumerator, axis = -1, keepdims = True)
    val = denumerator / numerator
    return val

class_prob = softmax(output) # 마지막 layer의 출력값을 classification을 위한 확률값으로 변환
-> array([[0.09003057, 0.24472847, 0.66524096],
          [0.04201007, 0.1141952 , 0.84379473],
          [0.57611688, 0.21194156, 0.21194156]])

[np.argmax(prob) for prob in class_prob] # 어떤 class가 가장 확률이 높은지
-> [2, 2, 0]
```

#### 3) 활성화 함수

활성화 함수는 선형 함수로부터 전달 받은 값을 출력할 때, 일정 기준에 따라 출력값을 변화시키는 비선형 함수입니다. 즉, 활성화 함수없이 딥러닝 모델을 구현하게 되면 선형 모델의 연속에 불과하게 됩니다.

전통적으로는 시그모이드(sigmoid)와 tanh 함수를 많이 사용했지만 ReLU 함수의 등장 이후로는, ReLU 함수를 통상적으로 사용합니다. 활성화 함수는 각 함수가 가지는 특징이 다르기 때문에 데이터와 모델에 따라 다양한 활성화 함수를 시도해보는 것이 좋은 성능을 이끌어낼 수 있습니다.

<image src = https://user-images.githubusercontent.com/48677363/106081630-8ae3c300-615c-11eb-9498-347faefba8ae.png width = 550>

  - **sigmoid: {0 <= a <= 1}**: z값이 너무 작거나 크게 되면, 기울기(slope)가 0에 가까워짐으로 gradient descent가 매우 천천히 진행되어 성능이 좋지 않아, output layer에서 주로 사용되는 함수임

  - **tanh: {-1 <= a <= 1}**: 대부분의 경우에 sigmoid보다 좋은 성능을 보이며 값의 범위가 -1과 +1 사이에 위치하게 되면서, 데이터 평균값이 0에 가깝게 유지되어 다음 층에서의 학습이 보다 수월함

  - **Relu: max(0, z)**: z가 0보다 클 때, 본래의 기울기를 가지는 특징으로 빠르게 gradient descent로 학습해 나갈 수 있기에, 가장 보편적으로 사용되는 함수임

  - **Leaky Relu**: Relu가 0의 값을 가질 때, 성능이 저하되는 것을 방지하기 위해 개선한 함수임



#### 4) 역전파 알고리즘

역전파 알고리즘인 **backpropagation**은 순전파 알고리즘을 통해 얻은 output과 실제 값과의 차이(loss)를 시작으로 각 층의 변화량을 기반하여 parameter를 반복적으로 업데이트하면서 loss를 줄여나가는 학습 방법입니다.

backpropagation을 이해하기 위해서는 순전파 알고리즘이 계산되는 과정과 역전파 알고리즘이 계산되는 과정을 그림과 코드로 하나하나 다루게 되면 보다 쉽게 이해할 수 있습니다.

먼저 다음과 같은 아주 간단한 구조의 Neural Network가 있습니다. 이 때, 입력되는 데이터와 각 층과 노드에서 계산되는 결과값을 직접 계산하면서 네트워크 진행과 학습 과정에 대해서 알아볼 수 있습니다.

![image](https://user-images.githubusercontent.com/48677363/115011033-17468c00-9ee9-11eb-94c4-fdb4d337038f.png)

**순전파 알고리즘**

위 그림에서 input layer에 입력되는 $x_1, x_2$ 벡터가 있습니다. 해당 입력 벡터와 각 입력에 따른 결과값을 다음과 같이 가정해볼 수 있습니다. 이 때, 벡터의 차원을 마음대로 설정할 수 있지만 여기서는 3차원으로 표현하였습니다.(이 때, '차원'이라는 단어 표현이 맞는지 모르겠습니다..ㅠ) 그리고 각 입력은 1과 0이라는 y값, 즉 정답을 가지고 있습니다. 결과적으로 우리는 주어진 입력 벡터를 인공신경망을 통해 정답을 맞춰가는 과정을 가지게 됩니다.

```python
np.random.seed(1)
x = np.random.randn(3,2)
y = np.array([1, 0])

print(x)

-> [[ 1.62434536 -0.61175641]
    [-0.52817175 -1.07296862]
    [ 0.86540763 -2.3015387 ]]
```

다음으로 주어진 입력 벡터가 네트워크를 통과하면서 계산이 이뤄집니다. 이 때, 각 네트워크의 층에서는 선형 계산이 이뤄지기 때문에 가중치(weight)와 bias가 필요합니다. 여기서는 bias는 제외하고 weight만으로 선형 계산이 이뤄지는 것을 살펴보겠습니다.

먼저 weight를 초기화(initialize)해야합니다. weight를 초기화하는 방법은 여러가지가 있지만 여기서는 랜덤 생성하도록 하겠습니다. hidden layer와 output layer에서 2번의 선형 계산이 발생하기 때문에 2개의 weight와 bias가 생성되어야 합니다.



```python
def initialize_parameters(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

layers_dims = [x.shape[0], 2, 1]
parameters = initialize_parameters(layers_dims)
print(parameters)

-> {
    'W1': array([[ 1.46040903,  0.3564088 ,  0.07878985],
                 [-1.52153542, -0.22648652, -0.28965949]]),
    'b1': array([[0.],[0.]]),
    'W2': array([[-0.08274148, -0.62700068]]),
    'b2': array([[0.]]
    }
```

weight와 bias가 parameter로 생성되었다면, 이제 네트워크를 통과하면서 계산되는 과정을 가지게 됩니다. 네트워크는 위에서 알아본 아주 간단한 형태로 구현할 수 있습니다. 이 때, 입력 벡터가 각각 계산되는 것이 아니라 행렬 계산인 vetorization이 적용되기 때문에 한번에 계산할 수 있습니다. 계산이 이뤄지는 과정이 한번에 이해가 되지 않을 수 있을 때에는, 각 계산의 결과를 출력하면서 어떠한 차원의 형태로 구성되어 있는지 확인하면 보다 쉽게 이해가 가능합니다.

이렇게 주어진 입력 벡터 $x$가 forward_propagation을 한번 통과하게 되면 1 epoch의 반이 완료되었다고 생각할 수 있습니다. 1 epoch에는 한번의 순전파 그리고 loss 계산, 역전파 그리고 업데이트의 과정을 가집니다.

코드를 자세하게 살펴보겠습니다. 먼저 첫번째 hidden layer를 통과하게 되면 2개의 결과가 나오게 되는데, 해당 결과는 첫번째 layer의 결과라고 하여, 각각 $a^{[1]}_1, a^{[1]}_2$라고 하며 통합적으로 $a^{[1]}$라고도 표현합니다. 즉, 코드 부분의 $a1$과 $a^{[1]}$은 같은 값을 저장하고 있습니다. 다음 layer에서는 출력값을 계산하는 마지막 layer로 0과 1의 logistic 계산을 위해 sigmoid 함수를 취해주게 됩니다. 결론적으로, $a2$는 $a^{[2]}_1, a^{[2]}_2$를 모두 포함하는 각 입력 벡터의 예측값을 의미합니다.
```python
def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return

def relu(x):
    s = np.maximum(0, x)
    return s

def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2)
    
    return a2, cache
```

**역전파 알고리즘**

위와 같이 순전파가 진행되었다면, 역전파가 진행될 차례입니다. 역전파에서는 순전파에서 예측한 예측값과 실제값의 차이를 줄여나가기 위해 발생한 차이를 계산하고 각 층에서 발생하는 오차를 모두 반영해야 합니다.

이 때, 각 층에서 발생하는 오차, 즉 미분값을 나중에 parameter 업데이트를 위해서 저장해줄 수 있습니다. 해당 과정은 한번의 순전파 과정에서 발생한 예측값의 차이를 줄여주기 위한 한번의 역전파 과정입니다.



```python
def backward_propagation(X, Y, cache):
    
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2) = cache
    
    dz2 = 1/m * (a2 - Y) # 예측값 - 실제값
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis = 1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis = 1, keepdims = True)
    
    gradients = {"dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients
```

예측값이 실제값에 보다 가까워지기 위해서는 parameter를 조정해주어야 합니다. 왜냐하면 본래 입력되는 입력 벡터는 고정되어 있는 상태이기 때문에 인공신경망에서 부여한 weight와 bias만이 변할 수 있기 때문입니다. 저는 이러한 과정을 주어진 어떠한 모양의 도형에 맞는 틀을 찾아가는 과정이라고 이해했습니다.

여기서 각 미분값에 따라 parameter가 조정될 때, learning rate가 미분값에 곱해지는 것을 볼 수 있습니다. 식 자체에서 직관적으로 알 수 있듯이, learning rate는 얼마정도의 오차로 parameter를 업데이트 해줄 것이냐를 결정해주는 hyperparameter입니다. 즉 1과 가깝다면, 계산된 오차만큼 업데이트 될 것이고 0.0001이라면 오차 * 0.0001만큼 업데이트 될 것입니다. 보통 모델 학습에 있어 learning rate를 항상 설정해주어야 하며 learning rate가 성능에 굉장히 많은 영향을 끼친다고 하기 때문에 자세히 알아보는 시간을 가지는 것이 좋습니다.

```python
def update_parameters_with_gd(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # 층의 깊이를 의미함

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
         
    return parameters
```


```python
def compute_cost(a3, Y):
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost_total =  np.sum(logprobs)
    
    return cost_total

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    
    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    
    parameters = initialize_parameters(layers_dims)
    print(parameters)
    for i in range(num_epochs):
        
        cost_total = 0
        
        a3, caches = forward_propagation(X, parameters)
        
        cost_total += compute_cost(a3, Y)
        
        grads = backward_propagation(X, Y, caches)
        
        parameters = update_parameters_with_gd(parameters, grads, learning_rate)
        
        cost_avg = cost_total / m
        
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
            
    return parameters
```

