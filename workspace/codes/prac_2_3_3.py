#%matplotlib inline
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

def XOR(*x) -> np.ndarray:
    if len(x) == 2: x = np.array([x[0], x[1]])
    return AND(NAND(x),OR(x))

if __name__=="__main__":
    x_inpts = [[0, 0]
              ,[1, 0]
              ,[0, 1]
              ,[1, 1]]
    results = [[x[0], x[1], AND(x), NAND(x), OR(x), XOR(x)] for x in x_inpts]
    print(pd.DataFrame(results, columns=["x1", "x2", "AND", "NAND", "OR", "XOR"]))
#     print(" AND(0, 0) =",AND(0, 0),"\n" # 0を出力
#          ,"AND(1, 0) =",AND(1, 0),"\n" # 0を出力
#          ,"AND(0, 1) =",AND(0, 1),"\n" # 0を出力
#          ,"AND(1, 1) =",AND(1, 1),"\n" # 1を出力
#     )
#     print(" NAND(0, 0) =",NAND(0, 0),"\n" # 1を出力
#          ,"NAND(1, 0) =",NAND(1, 0),"\n" # 1を出力
#          ,"NAND(0, 1) =",NAND(0, 1),"\n" # 1を出力
#          ,"NAND(1, 1) =",NAND(1, 1),"\n" # 0を出力
#     )
#     print(" OR(0, 0) =",OR(0, 0),"\n" # 0を出力
#          ,"OR(1, 0) =",OR(1, 0),"\n" # 1を出力
#          ,"OR(0, 1) =",OR(0, 1),"\n" # 1を出力
#          ,"OR(1, 1) =",OR(1, 1),"\n" # 1を出力
#     )

# #メモ 一次関数であらわされる場合は上記の通り。詳しくは参考書。以下は組み合わせで表現

#     print(" XOR(0, 0) =",XOR(0, 0),"\n" # 0を出力
#          ,"XOR(1, 0) =",XOR(1, 0),"\n" # 1を出力
#          ,"XOR(0, 1) =",XOR(0, 1),"\n" # 1を出力
#          ,"XOR(1, 1) =",XOR(1, 1),"\n" # 0を出力
#     )
