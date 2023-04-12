# Lite-Mono-ONNX-Sample
単眼深度推定モデルの[Lite-Mono](https://github.com/noahzn/Lite-Mono)のPythonでのONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。変換自体を試したい方は[Lite-Mono-Convert2ONNX.ipynb](Lite-Mono-Convert2ONNX.ipynb)を使用ください。<br>

https://user-images.githubusercontent.com/37477845/231474999-ea69d150-2dbf-4840-b276-d30441ecd2b2.mp4

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.13.0 or later

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/lite-mono-tiny_640x192

# Reference
* [noahzn/Lite-Mono](https://github.com/noahzn/Lite-Mono)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Lite-Mono-ONNX-Sample is under [MIT License](LICENSE).
