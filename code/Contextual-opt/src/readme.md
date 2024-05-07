# Readme
## ディレクトリ構成
```
.
├── experiment_template
├── model
│   ├── besier (未実装)
│   └── gp
├── objective_function
│   └── constraint_parameter (未実装)
├── optimizer
├── test_template
└── util
```
- __experiment_template__：実験用のテンプレート（新しい実験はこちらをコピーしてください）
    - `experiment_train.py`：学習用スクリプト
        - optimal response（最適解）を推定するガウス過程回帰の学習
        - 簡易的な評価関数`simplified_evaluation`での評価
        - 学習後にモデルを保存
    - `experiment_evaluate.py`：評価用スクリプト（未実装）
        - 後々必要に応じて実装（GDX IGDXなど）

- __model__：回帰モデルのクラス用ディレクトリ
    - `besier`：ベジエ単体用（未実装）
    - `gp`：ガウス過程回帰用（一部ベイズ最適化用）

- __objective_function__：目的関数のクラス用ディレクトリ
    - `base.py`：パラメータ付きの目的関数のベースクラス
    - `benchmark.py`：ベンチマーク用クラス
    - `constraint_parameter`：パラメータに依存した制約をもつ制約付き最適化用のクラス（未実装）

- __optimizer__：目的関数のクラス用ディレクトリ
    - `base_optimizer.py`：最適化法のクラスのベースクラス
    - `cmaes.py`：CMA-ES用クラス

- __test_template__：各スクリプトの動作確認用のテンプレート
    - `test_cmaes.py`：CMA-ESのテスト用
    - `test_gaussian_process.py`：ガウス過程回帰，ベイズ最適化のテスト用
    - `test_multiout_gaussian_process.py`：多出力のガウス過程回帰のテスト用

- __util__：その他関数（主にCMA-ES用）

## 実行方法
- 必要なライブラリのインストール
```
cd src
pip install -r requirements.txt
```
- `experiment_template/experiment_train.py`の実行の場合
```
cd src
python experiment_template/experiment_train.py
```
- `test_template/test_cmaes.py`の実行の場合
```
cd src
python test_template/test_cmaes.py
```

## TODO
### 実験の設定用のconfig fileの作成
- 評価回数の上限などの**実験設定**をjson形式で指定
- 参考：https://qiita.com/kohbis/items/f3156f822bac330494fd
### 実験ログを取るスクリプトの実装
- 評価指標の推移，評価した解とパラメータなどを保存
- config fileの保存も
### ベンチマーク関数の種類を増やす
- 現状のベンチマークでは最適解のみパラメータに線形に依存
- 関数形状がパラメータに依存するベンチマークの構成
- sin,cosなどの非線形変換を用いたベンチマークの構成
- `objective_function/benchmark.py`のクラスを参考
``` python
class NewBenchmark(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, 
                    d,                      # 次元数
                    max_eval=np.inf,        # 最大評価回数
                    param=None,             # パラメータ
                    param_range = [[-2,2]]  # パラメータの定義域
                ):
        super(NewBenchmark, self).__init__(max_eval, param, param_range)
        self.d = d          # 問題の次元数
        self.param_dim = 1  # パラメータの次元数

    def _evaluation(self, X, param):
        # (X, param)の評価値を返す
        return None
    
    def optimal_solution(self, param):
        # パラメータに対応する最適解（テスト用）
        # 計算ができなければNoneを返す
        return None
    
    def optimal_evaluation(self, param):
        # パラメータに対応する最良評価値（テスト用）
        # 計算ができなければNoneを返す
        return None
```
### ガウス過程回帰のカーネル，獲得関数の変更
- `experiment_template/experiment_train.py`を参考（下記の部分でカーネルは変更可能）
``` python
kernels_list = [RBF(f.param_dim), Linear(f.param_dim), Matern52(f.param_dim)]
```
- 獲得関数の変更は後回しでも
    - 現在は予測の共分散行列の行列式を獲得関数に利用（`model/gp/bo_acq.py`を参考）
### 実問題の実装・評価
- パラメータに依存した制約をもつ制約付き最適化
- 機械学習のハイパーパラメータ最適化
    - パラメータ：データセットの特徴量