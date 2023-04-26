# gpquant
## Introduction
As "genetic programming for quant", gpquant is a modification of the genetic algorithm package gplearn in Python, used for factor mining.

## Modules
### Function
Functions that calculate factors are implemented using the functional class `Function`, which includes 23 basic functions and 37 time series functions. All functions are essentially scalar functions, but because vectorized computation is used, both inputs and outputs are in vector form.

### Fitness
Fitness evaluation indicators are implemented using the functional class `Fitness`, which includes several fitness functions, mainly the Sharpe Ratio ("sharpe_ratio").

### Backtester
The vectorized factor backtesting framework follows the logic of first using the defined strategy function to turn the received "factor" into a "signal", and then using the signal processing function to turn the signal into an "asset" to implement backtesting. These two steps are combined in the functional class `Backtester`.

### SyntaxTree
The formula tree is used to write the calculation formula of the factor in prefix notation, and is represented using the formula tree `SyntaxTree`. Each formula tree represents a factor, and is composed of `Node`'s; each `Node` contains its own data, parent node, and child nodes. The `Node`'s own data can be a `Function`, variable, constant, or time-series constant.

The formula tree can be crossed over subtree mutated, hoisted, point mutated or reproduced (logic can be referred to gplearn).

### SymbolicRegressor
It contains the symbolic regression class (`SymbolicRegressor`). `gpquant` essentially uses genetic algorithms to solve the symbolic regression problem, and defines some parameters during the genetic process, such as population size and number of generations.

## Usage
### Import
Download the gpquant package (pip install gpquant) and import the SymbolicRegressor class.

### Test
Like the example in `gplearn`, performing symbolic regression on $y=X_0^2 - X_1^2 + X_1 - 1$ with respect to $X_0$ and $X_1$ can yield the correct answer at around the 9th generation.

```Python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import *
from gpquant.SymbolicRegressor import SymbolicRegressor


# Step 1
x0 = np.arange(-1, 1, 1 / 10.0)
x1 = np.arange(-1, 1, 1 / 10.0)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color="green", alpha=0.5)
plt.show()

# Step 2
rng = check_random_state(0)

# training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
X_train = pd.DataFrame(X_train, columns=["X0", "X1"])
y_train = pd.Series(y_train)

# testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

# Step 3
sr = SymbolicRegressor(
    population_size=2000,
    tournament_size=20,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutate=0.1,
    p_hoist_mutate=0.1,
    p_point_mutate=0.05,
    init_depth=(6, 8),
    init_method="half and half",
    function_set=["add", "sub", "mul", "div", "square"],
    variable_set=["X0", "X1"],
    const_range=(0, 1),
    ts_const_range=(0, 1),
    build_preference=[0.75, 0.75],
    metric="mean absolute error",
    parsimony_coefficient=0.01,
)

sr.fit(X_train, y_train)

# Step 4
print(sr.best_estimator)
```

# gpquant
## 介绍
gpquant是对Python的遗传算法包[gplearn](https://gplearn.readthedocs.io/en/stable/)的一个改造，用于进行因子挖掘
## 模块
### Function
计算因子的函数，用仿函数类Function实现了23个基本函数和37个时间序列函数。所有的函数本质上都是标量函数，但因为采用了向量化计算，所以输入和输出都是向量形式
### Fitness
适应度评价指标，用仿函数类Fitness实现了几个适应度函数，主要是应用其中的夏普比率sharpe_ratio
### Backtester
向量化的因子回测框架，逻辑是先根据定义的策略函数把拿到的因子factor变成信号signal，再通过信号处理函数把信号signal变成资产asset实现回测，这两步统一在仿函数Backtester类里实现
### SyntaxTree
公式树，把因子的计算公式写成前缀表达式，然后用公式树SyntaxTree表示。每一个公式树代表一个因子，由节点Node构成；每个Node存放了自身数据、父节点和子节点。节点的自身数据可以是Function、变量、常量，或者时间序列常数

公式树可以交叉crossover、子树突变subtree_mutate、提升突变hoist_mutate、点突变point_mutate或者繁殖reproduce（逻辑可参照gplearn）
### SymbolicRegressor
符号回归类，gpquant因子挖掘本质上是用遗传算法解决符号回归问题，其中定义了遗传过程中的一些参数，如种群数量population_size、遗传代数generations等

## 使用
### 导入
下载gpquant包（pip install gpquant），导入SymbolicRegressor类
```Python
from gpquant.SymbolicRegressor import SymbolicRegressor
```
### 测试
跟gplearn一样的例子，把$y=X_0^2 - X_1^2 + X_1 - 1$对$X_0$和$X_1$进行符号回归，大约在第9代能找到正确答案
```Python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import *
from gpquant.SymbolicRegressor import SymbolicRegressor


# Step 1
x0 = np.arange(-1, 1, 1 / 10.0)
x1 = np.arange(-1, 1, 1 / 10.0)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color="green", alpha=0.5)
plt.show()

# Step 2
rng = check_random_state(0)

# training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
X_train = pd.DataFrame(X_train, columns=["X0", "X1"])
y_train = pd.Series(y_train)

# testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

# Step 3
sr = SymbolicRegressor(
    population_size=2000,
    tournament_size=20,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutate=0.1,
    p_hoist_mutate=0.1,
    p_point_mutate=0.05,
    init_depth=(6, 8),
    init_method="half and half",
    function_set=["add", "sub", "mul", "div", "square"],
    variable_set=["X0", "X1"],
    const_range=(0, 1),
    ts_const_range=(0, 1),
    build_preference=[0.75, 0.75],
    metric="mean absolute error",
    parsimony_coefficient=0.01,
)

sr.fit(X_train, y_train)

# Step 4
print(sr.best_estimator)
```
