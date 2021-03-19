# Week08 - 모델 경량화

## [Day 36] - 

### 1. 모델 경량화 - 가벼운 모델

이번 강의에서는 Lightweight modeling techniques 를 이해하기 위한 Background와 Bird view를 제시합니다.

#### 1) 결정, Decision making

결정에는 연역적(Deductive) 결정과 귀납적(Inductive) 결정이 있습니다. **연역적(Deductive) 결정**이란 전제된 정의를 바탕으로 사례를 증명하는 논리를 말합니다. **귀납적(Inductive) 결정**은 수많은 사례를 기반으로 주장을 증명하는 논리입니다.

무언가를 결정해 나가는 **Decision making** 과정은 머신러닝 모델의 핵심이며 본질이라고 합니다.

**결정기, Decision making machine**

머신러닝 모델은 하나의 **결정기**라고 할 수 있습니다. 즉, 머신러닝 모델은 사람이 결정해야할 의사결정 문제를 대신할 수 있는 도구가 될 수 있음을 의미합니다. 하지만 모든 결정에는 그에 따른 책임이 존재하기 때문에 모든 결정을 머신러닝 모델로 대체할 수는 없습니다.

**가벼운 결정기, Lightweight Decision making**

***Lightweight*** 는 단순히 Light, 가벼운 것과 약간의 의미 차이가 존재합니다. 

