# 実験　CotextualCMA-ESとの比較
研究テーマについては[こちら](https://github.com/shiralab/sekino-yuta/blob/main/reserch-theme/reserch-theme.pdf)をご覧ください
## 実験方法
- 同じ関数について最適化（提案手法については既存最適解を与える）
- ランダムに選択した５つの文脈ベクトルについてそれぞれ評価値を算出し平均を取る
- この試行を10回行って平均を算出
- 全てのベンチマーク関数の最良評価値は０

## 実験結果
### 実験１
- **実験設定**
    - 関数：ShiftSphere

    $$f(\boldsymbol{x},\boldsymbol{\alpha})=\sum_{i=1}^{n}\boldsymbol{y}_i^2$$

    $$\boldsymbol{y}=\boldsymbol{x}-\boldsymbol{G}\boldsymbol{\alpha}$$

    $\boldsymbol{G}$ は平均0標準偏差1のガウス分布によってサンプリングされる{設計変数次元数×文脈ベクトル次元数}の定数行列

    - 次元数：20
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[-2~2]
    - 既存最適解数：3
- **実験結果**
    |          | 提案手法 | 既存手法 | 
    | -------- | -------- | ------ | 
    | **評価値** | **1.371e-07** | 1.357e-06 |  <br>

- **考察**
    - 若干提案手法の方が良い結果になったがそこまで変わらなかった
    - 今回の関数は文脈ベクトルの次元も低く，文脈ベクトルに対して線型に動いているため予測が簡単であった
    - 文脈ベクトルに対して非線形に最適解が動く場合の実験も行う

### 実験２
- **実験設定**
    - 関数：NonLinearShiftSphere
    - 次元数：20
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[0~2]
    - 既存最適解数：3

    NonLinearShiftSphere

    $$f(\boldsymbol{x},\boldsymbol{\alpha})=\sum_{i=1}^{n}\boldsymbol{y}_i^2$$

    $$
    \boldsymbol{y}=\boldsymbol{x}-\boldsymbol{G}\left(
    \begin{matrix}
    {\alpha_1}^2 \\
    {\alpha_2}^2 \\
    ... \\
    {\alpha_i}^2

    \end{matrix}
    \right)
    $$



- **実験結果**
    |          | 提案手法 | 既存手法 | 
    | -------- | -------- | ------ | 
    | **評価値** | **2.842e-04** | 6.211 |  <br>

- **考察**
    - 既存手法では評価値が大きく悪化したのに対して，提案手法は悪化が抑えられていた
    - 提案手法の強みの部分を確認することができた


### 実験３
- **実験設定**
    - 関数：ShiftSphere
    - 次元数：20
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[-2~2]
    - 既存最適解数：変更しながら実験
- **実験結果**
    | 既存最適解数 | ５ | ４ | ３ | ２ |
    | -------- | -------- | ------ | ------ | ------ | 
    | **評価値** | 4.5832e-14 | 1.1710e-13 | 2.3644e-05 | 3.2237e-4|  <br>

- **実験設定**
    - 関数：NonLinearShiftSphere
    - 次元数：20
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[-2~2]
    - 既存最適解数：変更しながら実験
- **実験結果**
    | 既存最適解数 | ５ | ４ | ３ | ２ 
    | -------- | -------- | ------ | ------ | ------ | 
    | **評価値** | 3.3711e-8 | 8.5331e-08 | 1.2945e-05 | 1.8199e-4|  <br>

- **実験設定**
    - 関数：NoisedShiftSphere
    - 次元数：20
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[-2~2]
    - 既存最適解数：変更しながら実験
- **実験結果**
    | 既存最適解数 | ５ | ４ | ３ | ２ |
    | -------- | -------- | ------ | ------ | ------ | 
    | **評価値** | 3.1094e-05 | 1.7967e-06 | 4.9027e-06 | 1.3963e-4|  <br>

### 実験４
- **実験設定**
    - 関数：Sphere,Rastrigin,Easom
    - 移動：Shift,NonlinearShift,NoisedShift
    - 次元数：20（Easomは次元数２）
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[-2~2]
    - 既存最適解数：5
- **実験結果**
    |          | Sphere   |          |          | Rastrigin|          |          | Easom    |          |          | Rosenbrock    |          |          |
    | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
    |          |Shift|NonlinearShift|NoisedShift|Shift|NonlinearShift|NoisedShift|Shift|NonlinearShift|NoisedShift|Shift|NonlinearShift|NoisedShift|
    | **提案手法** | 8.5364e-09 | 1.31678e-06 | 3.3359e-05 | 8.5084e-09 | 16.0335 | 14.4868 | 5.4078e-09 | 5.9993e-02 | 1.9998e-02 | 20.5397 |5.9993e-02 | 1.9998e-02 | 20.5397 |
    | **既存手法** | 2.3291e-06  | 5.7995 | 16.5142 | 218.2077 | 236.2499 | 255.8886 | 0.19449 | 0.72276 | 0.81208 | 20.5397 | 1.9998e-02 | 9227.2727 |<br>
    | **提案手法（単位行列スタート）** | 8.528117221114872e-09 | 8.593271319898045e-09 | 8.468681615532264e-09 | 8.780895797144694e-09 | 45.9044 | 23.2478 | 5.23266166396752e-09 | 5.106624576534102e-09 | 0.035493648371920636 |

### 実験５
- **実験設定**
    - 関数：Sphere,Rastrigin,Easom
    - 移動：Shift,NonlinearShift,NoisedShift
    - 次元数：20（Easomは次元数２）
    - 文脈ベクトル次元数：2
    - 文脈ベクトル範囲：各要素[-2~2]
    - 既存最適解数：5
- **実験結果**<br>
    Shift<br>
    
    ![結果](https://github.com/shiralab/sekino-yuta/blob/main/experiment-result/ContextualCMAES/cma_sphere01.png)

    NolinearShift<br>

    ![結果](https://github.com/shiralab/sekino-yuta/blob/main/experiment-result/ContextualCMAES/cma_sphere02.png)

    NoisedShift<br>

    ![結果](https://github.com/shiralab/sekino-yuta/blob/main/experiment-result/ContextualCMAES/cma_sphere03.png)

    |          | Sphere   |          |          | 
    | -------- | -------- | -------- | -------- |
    |          |Shift|NonlinearShift|NoisedShift|
    | **提案手法** | 8.4619e-09 (1332.8) | 4.9273e-08(9793.6) | 3.2463e-05(9815.2) |
    | **CMA-ES** | 8.39124e-09 (2659.2)  | 8.1614e-09(2720.8) | 8.3601e-09(2656.8) |<br>

