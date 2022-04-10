---
image: https://images.credly.com/size/680x680/images/5c8fca38-b0d2-49e5-9ad2-f3f8e79b327f/azure-data-scientist-associate-600x600.png
tags: DP-100, Note
GA: UA-117096964-1
---

# DP-100 NOTE

## M08 - Hyperparameter
透過不同超參數值和同一個算法來得到不同模型，然後取得`最佳`解

### Discrete Hyperparameter (離散)
=> `choice`
- quniform(low, high, q) => round(uniform(low, high) / q) *q
- qloguniform => exp(quniform(low, high))
- qnormal(mu, sigma)
- qlognormal => exp(qnormal(mu, sigma))

``` python
{
    "batch_size": choice(16, 32, 64, 128)
    "number_of_hidden_layers": choice(range(1,5))
}
```

### Continuous hyperparameters (連續)
- uniform(low, high) - return low ~ high 
- loguniform => exp(uniform(low, high))
- normal(mu, sigma) - mu => 平均值，sigma => 標準差，所得到的實數常態分佈
- lognormal => exp(normal(mu, sigma)) => 

``` python
{    
    "learning_rate": normal(10, 3),
    "keep_probability": uniform(0.05, 0.1)
}
```

### Sampling the Hyperparameter space
- Random Sampling
- Grid Sampling
- Bayesian Sampling

#### Random Sampling
Support Disceret and Continuous Hyperparameter

在定義的範圍之內，隨機選取 Value

``` python
from azureml.train.hyperdrive import RandomParameterSampling
from azureml.train.hyperdrive import normal, uniform, choice
param_sampling = RandomParameterSampling( {
        "learning_rate": normal(10, 3),
        "keep_probability": uniform(0.05, 0.1),
        "batch_size": choice(16, 32, 64, 128)
    }
)
```

#### Grid Sampling
Support Disceret Hyperparameter only

嘗試所有離散的組合

``` python
from azureml.train.hyperdrive import GridParameterSampling
from azureml.train.hyperdrive import choice
param_sampling = GridParameterSampling( {
        "num_hidden_layers": choice(1, 2, 3),
        "batch_size": choice(16, 32)
    }
)
```

#### Bayesian Sampling
Base on Bayesian algorithm 

``` python
from azureml.train.hyperdrive import BayesianParameterSampling
from azureml.train.hyperdrive import uniform, choice
param_sampling = BayesianParameterSampling( {
        "learning_rate": uniform(0.05, 0.1),
        "batch_size": choice(16, 32, 64, 128)
    }
)
```

### Primary mertic to optimize

- primary_metric_name
- primary_metric_goal

``` python
primary_metric_name = "accuracy",
primary_metric_goal = PrimaryMetricGoal.MAXIMIZE
```

#### Log metric

``` python
from azureml.core.run import Run
run_logger = Run.get_context()
run_logger.log("accuracy", float(val_accuracy))
```

### Early termination policy
- Bandit 原則
- 中位數停止原則
- 截斷選取原則
- 無終止原則 (Default)

#### Bandit policy
当指标与迄今为止的最佳运行在性能方面达到指定差距时停止

- slack_facor or slack_amout => 為取得效能最佳之定型執行所允許的寬限時間
- evaluation_interval => 套用原則的頻率 (Optional)
- delay_evaluation => 延遲多久才開始進行評估 (Optional)

``` python
from azureml.train.hyperdrive import BanditPolicy
early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
# 1/(1+0.1) = 91%
```

#### Median Stopping policy
在指标低于运行平均值的中值时停止
- evaluation_interval => 套用原則的頻率 (Optional)
- delay_evaluation => 延遲多久才開始進行評估 (Optional)

``` python
from azureml.train.hyperdrive import MedianStoppingPolicy
early_termination_policy = MedianStoppingPolicy(evaluation_interval=1, delay_evaluation=5)
```

#### Truncation Selection policy
在指标处于同一间隔中所有运行最差的 X％ 中时停止

``` python
from azureml.train.hyperdrive import TruncationSelectionPolicy
early_termination_policy = TruncationSelectionPolicy(evaluation_interval=1, truncation_percentage=20, delay_evaluation=5, exclude_finished_jobs=true)
```

#### 無終止原則 (Default)
``` python
policy = None
```

#### How to choice Early termination policy
- Save cost => Median Stopping policy `MedianStoppingPolicy(evaluation_interval=1, delay_evaluation=5)` => save 25% ~ 35% cost
- Save more cost => Bandit policy

## M09 - Responsible AI

### Principle
![](https://mdcontent.yu.money/contents/upload_0198dc6ca4bb99ca3851f1b6bcaac565.png)

- Understand
    - 解讀模型的行為
    - 評估一些不適當之處
- Protect
    - 防止隱私曝光
    - 加密處理
- Control
    - 用表格來記錄 ML 的過程或是生命週期

#### 解讀模型的行為
量化每个特征对预测的影响
对于识别模型中的偏差或非预期相关性很重要

##### Azureml-interpret (open-source package)
- MimicExplainer - 近似你的模型的全局代理模型
- TabularExplainer - 基于模型架构调用直接的 SHAP 解释器
- PFIExplainer - 基于特征变换的排列特征重要性

#### 差別隱私
![](https://mdcontent.yu.money/contents/upload_2b518bb53b2bcc2a5b7fff9bcc2798b5.png)

防止無限制的存取
- epsilon => 來測量報告的雜訊或者是隱私程度 => epsilon 跟雜訊或隱私程度反比關係 => best: 0~1
- delta => 報告非完全私用的機率值
- epsilon & delta 互相關聯的

#### Privacy Budget
定義速率限制
通常會與 epsilon 有關聯，通常是 1~3 => 用來限制重新識別的風險

#### 資料可靠性
必須在隱私權以及資料的可用與可靠性之間去做一個取捨
=> 較高層級的雜訊帶來的就是較低的 epsilon、準確度和可靠性的資料

### 公平性
#### 缓解不公平性
使用奇偶校验约束创建模型：
- 人口统计奇偶校验：最大限度减小敏感特征组之间的选择率差异。
- 真正例比率奇偶校验：最大限度减小敏感特征组之间的真正例比率差异
- 假正例比率奇偶校验：最大限度减小敏感特征组之间的假正例比率差异
- 均等几率：最大限度减小敏感特征组之间的合并的真正例比率和假正例比率的差异
- 错误率奇偶校验：确保每个敏感特征组的错误偏离总体错误率的程度不超过指定数量
- 有界组损失：限制回归模型中每个敏感特征组的损失

## M10
### Dataset - data drift
不断变化的数据趋势，可能会影响已训练模型的准确度
- 上游程序變更
- 資料品質
- 資料的天然變化性
- 變項飄移