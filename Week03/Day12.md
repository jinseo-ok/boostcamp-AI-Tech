# Week03 - DL Basic

## [Day 12] - 최적화

### 1. Optimization

#### 1) Optimization의 기본 용어

  - **Generalization**

보통 딥러닝에서는 **Generalization**, 일반화 성능을 높이는 것을 목적으로 합니다. 사실 머신러닝과 딥러닝을 공부하는 초반에는 이 generalization의 목적과 개념을 이해하기 어려웠습니다. 주어진 데이터로 하여금 모델의 성능을 높이는 것이 머신러닝과 딥러닝 분야의 목적 및 핵심이라고 생각했기 때문입니다.

하지만 모델의 성능을 높이는 것은 AI 모델을 개발하고 서비스화하는 것에 있어 궁극적인 목표라기 보다는 하나의 과정이라고 생각하게 되었습니다. 사실 모델 학습을 위해 주어진 데이터는 모든 경우를 고려한 데이터가 아닐 가능성이 높습니다. 그렇기 때문에 테스트 데이터 셋을 별도로 구축하기는 하겠지만, 주어진 데이터로만 무작정 성능을 높이는 것은 숲을 보지 못하고 나무만을 보게 되는 것과 비슷하다고 생각했으며, 이를 오버피팅(Overfitting) 이라고 합니다. 그래서 보통 모델을 학습함에 있어, 보다 많은 데이터가 있다면 강건하고 성능이 높은 모델을 구축할 수 있다고 하는 것 같습니다.

결론적으로 딥러닝에서 Generalization은 굉장히 중요한 개념입니다. 주어진 데이터로 구축한 모델은 전혀 고려하지 않았던 테스트 데이터 셋과 새롭게 발생하는 데이터에 적용하게 됩니다. 우리는 관측하지 못했던 데이터에도 모델이 변함없이 작동하기 위해서 일반적인, 즉 general한 성능을 유지하도록 모델을 설계할 때 주의해야합니다.

train 데이터로 학습한 모델은 시간이 지나면 test 데이터에 대한 error가 커질 수 밖에 없습니다. 데이터를 학습함에 있어, 인공신경망, 즉 뉴런 혹은 노드들은 주어진 train 데이터에서만 비선형적인 패턴을 찾아나가기 때문입니다. generalization performance는 일반적으로 train 데이터 error와 test 데이터 error와의 차이를 의미합니다.

그러므로 예상치 못한 test 데이터를 예측함에 있어서도 error가 크게 발생하지 않는 견고한 모델은, generalization이 우수하다고 볼 수 있습니다.

<image src = https://user-images.githubusercontent.com/48677363/106565612-6def2b80-6572-11eb-887c-36ab32ddefda.png width = 500>

  - **Underfitting vs Overfitting**

**underfitting**과 **overfitting**은 train 데이터에 대한 모델의 학습 정도를 의미합니다. underfitting은 신경망의 깊이가 너무 얇거나, 학습 횟수가 적어 train 데이터 조차도 제대로 예측하지 못하는 경우를 의미하며, 반대로 overfitting은 모델이 train 데이터의 특징 혹은 패턴만을 고려하여 학습되어 test 데이터 혹은 다른 데이터 집합의 특징을 전혀 고려하지 못하는 경우를 의미합니다.

<image src = https://user-images.githubusercontent.com/48677363/106566384-94619680-6573-11eb-9ddf-6374f01ddc02.png width = 550>

  - **Cross-validation**

cross-validation(cv)는 교차 검증이라고 하며, 머신러닝과 딥러닝으로 문제를 해결함에 있어서 보통 적용하는 기법 중 하나입니다. 사실 머신러닝과 딥러닝을 공부하게 되면 캐글 대회를 참여하고 정리된 노트북을 보게 되는 경우가 많습니다. 그 때, 잘 정리된 노트북을 참고하게 되면 교차 검증이 성능 평가 시에 자주 사용되는 것을 확인할 수 있고, 해당 노트북을 참고하여 교차 검증을 적용해 볼 수 있습니다.
  
먼저 교차 검증은 generalization한 모델을 구축하기 위해서 본래 가지고 있던 train 데이터를 train, valid로 구분하여 학습을 진행하는 기법이라고 합니다. 이렇게 validation 과정을 거침으로써 보다 최적화된 hyperparameters들을 찾을 수 있으며, 이 때 가장 중요한 것은 test 데이터는 학습 과정에서 어떠한 방법으로도 사용되어서는 안된다는 것입니다. 다음은 교차 검증 과정에 대한 자료입니다.

<image src = https://user-images.githubusercontent.com/48677363/106567270-eb1ba000-6574-11eb-97a4-1805d7d656e9.png width = 500>

그러나 데이터 분석을 함에 있어, 교차 검증을 적용하는 과정에 많은 의문들을 마주치게 되었습니다. 다음은 제가 마주친 교차 검증에 대한 의문들입니다.
  
**Q. 교차 검증의 핵심 목표가 무엇인가?**

사실 교차 검증을 처음 진행하게 되면, valid 데이터에 대한 모델의 성능이 나오는 것이 굉장히 신기합니다. 하지만 단순히 성능 평가를 자주 하는 것이 모델의 generalization에 어떤 영향을 미치는지 의문점을 가지게 되었습니다. 먼저, 코드를 보게 되면 대부분, valid 데이터는 train 데이터의 일부를 선택하여 성능 평가에만 사용되는 것을 알 수 있습니다. 단순히 성능 평가에만 사용되는 valid 데이터가 모델의 일반화에 영향을 미칠 수 있을까라는 생각을 가지게 되었습니다. 그리고 이렇게 학습에 사용되는 데이터의 양이 줄어들면 오히려 모델의 학습에 부정적인 영향을 미치지 않을까 라는 생각이 들기도 합니다,
  
교차 검증에 대한 몇몇 블로그의 설명글을 보게 되면, 'test 데이터의 과적합(overfitting)을 방지하기 위해 교차 검증 기법을 사용한다.' 라는 것을 찾아볼 수 있습니다. 저는 개인적으로 해당 내용에 대해 정확히 이해하지 못했습니다. test 데이터는 모델의 학습 과정에서 어떠한 경우에도 사용되어서는 안된다고 알고잇는데 test 데이터의 과적합이 정확히 어떤 의미로 사용되었는지 이해하지 못해 댓글로 질문을 드리기도 했습니다.
  
다행히도 좀 더 블로그를 찾아보게 되면서 교차 검증의 목표에 대한 보다 이해가 되는 글을 찾을 수 있엇습니다.
  
  - [로스카츠의 AI 머신러닝, [머신러닝] 크로스 밸리데이션(cross validation, 교차 검증)의 개념, 의미](https://losskatsu.github.io/machine-learning/cross-validation/)
  
  - [Deep play, 딥러닝 모델의 K-겹 교차검증 (K-fold Cross Validation)](https://3months.tistory.com/321)

위 두 블로그의 내용을 정리해보자면 교차 검증의 목표는 valid 데이터로 모델의 성능을 평가함으로써 모델의 하이퍼파라미터를 설정 및 튜닝할 수 있다고 합니다. 사실 train 데이터만을 가지고 학습을 진행하게 된다면, 우리는 어떤 성능을 목표로 하이퍼파라미터를 튜닝해야할지 의문점을 가지게 됩니다. 예를 들어, train의 성능을 높이는 것으로만 하이퍼파라미터를 튜닝하게 되면 결국 overfitting이 되기 마련이기 때문입니다.
  
그렇기 때문에 valid 데이터라는 독립적인 데이터 셋을 구축하여 성능 평가에 활용함으로써, valid 데이터의 성능을 높이는 것을 목표로 하이퍼파라미터를 튜닝할 수 있습니다. 그렇다면 test 데이터를 valid 데이터처럼 성능 평가하여 하이퍼파라미터 튜닝에 지속적으로 사용하면 되는 것이 아닌가라는 의문이 생기기 마련입니다. 하지만 test 데이터는 말그대로 테스트를 위한 데이터 셋이므로 학습 과정에서 절대 사용해서는 안되며, 학습을 마친 모델의 성능을 평가하는 한번의 과정에서 사용하는 것을 습관으로 만드는 것이 중요할 것 같습니다.
  
**Q. 교차 검증 시, valid 데이터는 성능 평가 이외에도 학습 시 영향을 미치는가?**
  
해당 의문에 대해서는 어느정도 해결할 수 있었습니다. valid 데이터는 학습 과정에서 노드들의 가중치가 업데이트되는 것에는 전혀 영향을 미치지 않습니다. 교차 검증의 목표 자체가, 관측되지 않은 데이터에 대한 모델의 성능을 평가함으로써, 하이퍼파라미터를 튜닝해나가는 것이기 때문에 valid 데이터가 순전파와 역전파 과정에는 전혀 관여하지 않는다고 이해하게 되었습니다.
  
**Q. 교차 검증 시, valid 데이터는 train 데이터와 독립적인 데이터로 바라보아야 할 것 같은데, 데이터 스케일링을 독립적으로 적용해주어야 하지 않은가?**
  
해당 의문에 대해서는 여전히 진행 중입니다. 사실 교차 검증 기법을 사용할 때, 저는 sklearn의 model_selection의 K-Fold를 가장 자주 사용하였습니다. K-Fold를 사용하는 방법으로는, 구축된 모델로 하여금 train 데이터를 자동으로 train과 valid로 split하여 n번의 과정을 거치게 됩니다. 해당 과정은 데이터 전처리를 완료하고, 모델을 구축한 뒤 마지막 단계에서 이뤄지는 성능 평가입니다.
  
그런데 valid 데이터를 unseen 데이터로 취급하기 위해서는 전처리를 독립적으로 진행해줘야하지 않을까 라는 의문을 가지게 되었습니다. 사실 최근에는 데이터의 스케일을 맞춰주는 데이터 스케일링을 무조건적으로 시행하는 편은 아니지만, 머신러닝 모델에서는 여전히 연속적인 데이터에 데이터 스케일링을 진행해주는 과정이 필요하게 됩니다. 그렇다면 이 때 전처리 파이프라인을 설계할 때에 valid 데이터 셋에 대한 전처리 파이프라인을 독립적으로 따로 적용해주어야 하지 않을까라는게 제 생각이었습니다. 데이터 스케일링을 진행하게 되면 보통 적용하는 min-max scaler 혹은 StandardScaler는 전처리 시 적용되는 데이터 분포를 모두 확인하기 때문에 전처리 후에 train과 valid가 나뉘게 되면, valid 데이터는 이미 train의 데이터 분포와 특성이 반영되는 것이기 때문입니다.
  
하지만 위에 대한 의문을 제대로 다룬 다른 블로그를 찾기 못해, 여전히 정답 혹은 해답이 무엇인지 궁금해하고 있습니다. 해당 의문에 대해서 다룬 블로그 내용도 있습니다!
https://blog.naver.com/gdpresent/221730873049
  
  - **Bias and Variance**

모델의 에러를 최소화 하는 것은 bias와 variance의 trade-off를 의미합니다.

<image src = https://user-images.githubusercontent.com/48677363/106567400-1b633e80-6575-11eb-9598-84abf38cb939.png width = 300>

<image src = https://user-images.githubusercontent.com/48677363/106569971-9c700500-6578-11eb-9498-2e36602d6bb9.png width = 450>

  - **Bootstrapping** [참고자료](https://learningcarrot.wordpress.com/2015/11/12/%EB%B6%80%ED%8A%B8%EC%8A%A4%ED%8A%B8%EB%9E%A9%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC-bootstrapping/)

학습 데이터가 고정되어 있을 때, sub sampling을 통해 여러 개의 학습 데이터를 생성하고 여러 모델 및 평가지표를 적용해서 표본집단에서 보다 정확한 모수를 얻을 수 있습니다.

  - **Bagging vs Boosting**

**Bagginng(Boostrapping aggregating)** 은 sub sampling을 통한 다수의 학습 데이터에 다양한 모델을 적용하여 결과를 합쳐 하나의 새로운 결과를 출력하는 방법입니다. 이 때, 하나의 새로운 결과로 취합하는 과정에서는 평균 혹은 voting 등의 방법이 사용됩니다.

**Boosting**은  학습 데이터에 각 모델들을 독립적으로 적용하여 독립적인 결과를 취합하는 것이 아닌, 전체 학습 데이터에 모델들을 상호 취합하여 결과를 출력하는 방법입니다. 

<image src = https://user-images.githubusercontent.com/48677363/106571397-7186b080-657a-11eb-8a3e-4920a9c7d9af.png width = 500>

#### 2) Practical Gradient Descent Methods

  - Stochastic gradient descent: 데이터 1개씩(single sample) gradient를 구하면서 업데이트를 진행하는 방식

  - Mini-batch gradient descent: 전체 데이터 중 일부만을 가지고 gradient를 구해 업데이틑 진행하는 방식

  - Batch gradient descent: 전체 데이터를 모두 사용해 gradient를 구해 업데이트를 진행하는 방식

**Batch-size Matters**

hyperparameter 중 하나인 Batch size가 보통 64, 128, 256 등의 수치를 기본적으로 사용하는 편이지만 굉장히 중요한 hyperparameter입니다.

large batch size를 사용하게 되면 sharp minimizers에 도달하게 되고
small batch size를 사용하게 되면 flat minimizers에 도달하게 됩니다,

보통 small batch size를 적용하게 되면 test 데이터에 적용하게 되어도 보다 generalization된 결과를 얻을 가능성이 높습니다. (약간 learning rate와 비슷한 느낌을 받았음)

#### 3) Gradient Descent Methods

  - **Gradient Descent**

  - **Momentum**

  - **Nesterov Accelerated Gradient(NAG)**

  - **Adagrad**

  - **Adadelta**

  - **RMSprop**

  - **Adam**

#### 4) Regularization

  - **Early stopping**

  - **Parameter norm penalty**

  - **Data augmentation**

  - **Noise robustness**

  - **Label smoothing**

  - **Dropout**

  - **Batch Normalization**

---------

### 2. CNN 기초

Convolution 연산은 이미지 혹은 영상을 처리하기 위한 모델에서 통상적으로 사용됩니다. 이 전 강의에서 배웠던 MLP와 비교해서 CNN, Convolutional Neural Network의 커널 연산이 가지는 장점과 다양한 차원에서 진행되는 연산 과정에 대해서 배우게 됩니다.

Convolution 연산의 경우, 커널의 모든 입력 데이터에 대해 공통으로 적용이 되기 때문에 역전파를 계산하는 경우에도 똑같이 Convolution 연산이 출력됩니다. 강의에서 그림과 함께 잘 설명되어 있기 때문에 커널을 통해 gradient가 전달되는 과정과 역전파 과정에 대해 이해하고 있는 것이 좋습니다.

#### 1) Convolution 연산

이 전 강의에서 배운 다층신경망, MLP는 각 뉴런들이 선형모델과 활성함수로 모두 연결된, fully connected 구조 였습니다. 다층신경망은 입력데이터와 가중치 행렬의 i 위치와의 행렬곱을 통해서 계산이 이뤄지기 때문에 가중치 행렬의 구조가 복잡하고 커지게 되어 학습 파라미터가 많아지게 됩니다.

<image src = https://user-images.githubusercontent.com/48677363/106689647-adba1f80-6613-11eb-9fd5-c76bbe91e290.png width = 500>
<center> [ fully connected 연산 ] </center>

<br>
<br>

반면에, **Convolution 연산** 은 커널, Kernel 이라는 고정된 가중치를 사용해 입력벡터 상에서 움직이며 선형모델과 합성함수가 적용되는 구조입니다. 입력 데이터를 모두 동일하게 활용하는 것이 아니라 커널 사이즈에 대응되는 입력 데이터만을 추출하여 활용함을 의미합니다. 활성화 함수를 제외한 Convolution 연산도 선형변환에 해당합니다.

<image src = https://user-images.githubusercontent.com/48677363/106689976-4781cc80-6614-11eb-8f33-df1fcd78ab0f.png width = 500>
<center> [ convolution 연산 ] </center>

<br>
<br>

Convoltuion 연산의 수학적인 의미는 신호(signal)를 커널을 이용해 국소적으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것입니다. 일반적으로 CNN에서 사용되는 Convolution 연산은 구체적으로 보자면 빼기가 사용되지 않고 더하기가 사용된 cross-correlation 입니다. 커널은 정의역 내에서 움직여도 변하지 않고(translation invariant) 주어진 신호에 국소적(local)으로 적용됩니다.

#### 2) 2D Convolution 연산

2D Convolution 연산은 커널을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조입니다. 2차원 커널이 다음과 같이 주어져있을 때, 입력 데이터에는 커널의 크기만큼 element-wise 곱셈처럼 계산됩니다. 이 때, 입력 데이터에 커널이 이동하면서 계산이 이뤄지는데, 이동 범위에 해당하는 hyperparameter가 stride 입니다.

<image src = https://user-images.githubusercontent.com/48677363/106712012-b45c8d00-663b-11eb-881b-b9bf2c6f5542.png width = 600>

커널의 크기와 입력 데이터의 차원을 미리 알고 있다면, convolution 연산의 출력을 계산해볼 수 있습니다. 입력 데이터 크기를 ($H$, $W$), 커널 크기를($K_H$, $K_W$)라 한다면 출력 크기, ($O_H$, $O_W$) 를 다음과 같이 계산할 수 있습니다.
  
만약 28 * 28 input 데이터를 3 * 3 커널로 2D-Conv 계산을 하게 되면 output의 크기는 26 * 26 이 됩니다.

$$
O_H = H - K_H + 1
\\
O_w = W - K_w + 1
$$

또한 채널이 여러개인 2차원 이상의 데이터의 경우에는 2차원 Convolution을 채널 개수만큼 적용하게 됩니다. 결국 입력 데이터의 채널의 개수만큼 커널이 적용되어 계산이 이뤄진다면 출력 데이터는 1개의 채널의 2차원 데이터가 될 것입니다.

만약 출력 데이터 또한 채널의 개수 혹은 다차원이기를 원한다면 적용되는 커널의 차원 자체를 증가시켜 적용하게 되면 출력 데이터는 입력 데이터에 독립적으로 적용되는 커널의 개수만큼 나오게 됩니다.

<image src = https://user-images.githubusercontent.com/48677363/106714661-80836680-663f-11eb-94ae-30ce206cc7e7.png width = 600>

#### 3) Convolution 연산의 역전파

출력 데이터는 입력 데이터와 커널과의 element-wise 곱셉으로 계산된 데이터입니다. 그렇기 때문에 역전파 또한 convolution 연산처럼 커널 크기에 따라 진행이 됩니다.

<image src = https://user-images.githubusercontent.com/48677363/106716437-c93c1f00-6641-11eb-841f-9bc4e8f7cc34.png width = 500>
