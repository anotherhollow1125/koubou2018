
# 1 AND回路, NAND回路, OR回路の実装する

ⅰ. 関数名は各々，`AND(x1, x2)`, `NAND(x1, x2)`, `OR(x1, x2)`として，numpy配列を引数に取れるようにする。
(出力結果は[3](#3-出力結果)にて、[2](#2-1で実装した関数を利用してXOR関数を実装する)とまとめて行うこととする。)

関数名は若干変わってしまったが、numpy配列(np.ndarray型)を引数にとれる仕様としました。それ以外は
[赤本](https://www.oreilly.co.jp/books/9784873117584/)に準拠しました。


```python
%matplotlib inline
import pandas as pd
import numpy as np

def AND(*x) -> np.ndarray:
    if len(x) == 2: x = np.array([x[0], x[1]])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(*x) -> np.ndarray:
    if len(x) == 2: x = np.array([x[0], x[1]])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(*x) -> np.ndarray:
    if len(x) == 2: x = np.array([x[0], x[1]])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

# 2 1で実装した関数を利用してXOR関数を実装する

ⅱ. 関数名は，`XOR(x1, x2)`として，numpy配列を引数に取れるようにする。

こちらは少しだけ[赤本](https://www.oreilly.co.jp/books/9784873117584/)とは実装を変えました。numpy配列を引数としてとるために、比較的シンプルなコードとなりました。


```python
def XOR(*x) -> np.ndarray:
    if len(x) == 2: x = np.array([x[0], x[1]])
    return AND(NAND(x),OR(x))
```

# 3 出力結果


```python
x_inpts = [[0, 0]
          ,[1, 0]
          ,[0, 1]
          ,[1, 1]]
results = [[x[0], x[1], AND(x), NAND(x), OR(x), XOR(x)] for x in x_inpts]
pd.DataFrame(results, columns=["x1", "x2", "AND", "NAND", "OR", "XOR"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>AND</th>
      <th>NAND</th>
      <th>OR</th>
      <th>XOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


