## 動作環境
ProductName:            macOS

ProductVersion:         13.4

BuildVersion:           22F66

Python 3.9.16

## ディレクトリ構成
- __src__：実験用のコード
    - experiment_simulate：シミュレーションタスク実験用
        - `experiment_evaluate_py`:実験用スクリプト(CCMA-ES以外)（数字がついているのは並列実行用）
        - `ccmaes.py`:CCMA-ES実験用スクリプト（数字がついているのは並列実行用）
        - plot/`plot.py`:最良評価値推移グラフ作成用
        - plot/`plot_pena.py`:ペナルティ推移グラフ作成用
        - video:動画保存用
        - log:logファイル保存用
    - experiment_template：ベンチマーク関数実験用
        - `experiment_evaluate_py`:実験用スクリプト(CCMA-ES以外)
        - `ccmaes.py`:CCMA-ES実験用スクリプト
        - plot/`plot.py`:最良評価値推移グラフ作成用
        - plot/`plot_changeoptnum.py`:事前最適化回数を変更した時のグラフ作成用
        - log:logファイル保存用
- __gymnasium_robotics__：gymnasiumライブラリのコード
    - gymnasiumライブラリのコードを一部変更したため，gymnasiumライブラリをインストールした後同名のフォルダをこのフォルダで上書きする必要あり

## 実行方法
- 基本的な実行コマンド
```
cd Contextual-opt
python ファイル名
```
- 実験から結果出力までの流れ（ベンチマーク関数）
    - `ccmaes.py`を実行し，CCMA-ESの最良評価値の中央値，第一四分位数，第三四分位数を取得
    - 変数*Concmaeval,Concmaeval25,Concmaeval75*に取得した中央値，第一四分位数，第三四分位数を入れ，`experiment_evaluate.py`を実行し`plot/plot_log.csv`を生成
    - `plot/plot_log.csv`を実験内容に応じて名前変更
        - linearshiftのeasomだったら`plot_log_0_easom.csv`
        - nonlinearshiftのrosenbrockだったら`plot_log_1_rosenbrock`
        - noisyshiftのsphereだったら`plot_log_2_sphere.csv`
    - `plot/plot.py`を実行
        - 事前最適化数を変更したグラフは`plot/plot_changeoptnum.opt`を実行

- 実験から結果出力までの流れ（シミュレーション）
    - 事前に物理シミュレーターMujocoをインストール
    - `ccmaes.py`を実行し，CCMA-ESの最良評価値の中央値，第一四分位数，第三四分位数を取得
    - `experiment_evaluate.py`を実行
    - 変数*Concmaeval,Concmaeval25,Concmaeval75*に取得した中央値，第一四分位数，第三四分位数を入れ，`plot/make_plot_log.py`を実行し`plot/plot_log.csv`を生成
    - `plot/plot_log.csv`を実験内容に応じて名前変更
        - FetchPushの設計変数4次元だったら`plot_log_push_4dim.csv`
         - FetchSlideの設計変数3次元だったら`plot_log_slide_3dim.csv`
    - `plot/plot.py`を実行
        - ペナルティの推移を得る場合は`plot/plot_pena.py`を実行
