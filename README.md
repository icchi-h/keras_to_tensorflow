# keras_to_tensorflow
kerasで出力したmodelファイル(.h5)をtensorflowのmodel形式(.pb)に変換するスクリプト

[amir-abdi](https://github.com/amir-abdi)さんが公開されている.ipyenvコードを.pyに変換してから改良。  
<https://github.com/amir-abdi/keras_to_tensorflow>

* keras modelをコマンドライン引数から選択
* 出力ファイル名は入力ファイルを元に定義


## 使い方
**kerasとtensorflowが動く環境を用意**

```bash
pip install keras
pip install tensorflow
```

**以下のようなディレクトリ構造に**
> 分類器ディレクトリ(以下から取得)
> https://github.com/opencv/opencv/blob/master/data/haarcascades/
> https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades>/

```
.
├── README.md
├── keras_model
│   └── keras_model.h5
├── keras_to_tensorflow.py
└── tensorflow_model (存在しない場合は自動生成)
```

**実行**
```bash
# sample
python keras_to_tensorflow.py keras_model/keras_model.h5
```


# Original README
## keras_to_tensorflow
General code to convert a trained keras model into an inference tensorflow model

The notebook ```keras_to_tensorflow```, is a sample code which loads a trained keras model, freezes the nodes (converts all tensorflow variables to tensorflow constants), and saves the inference graph and weights into a protobuf file (.pb). This file can then be used to deploy the trained model for inference. During freezing, other nodes of the network, which do not contribute  the tensor that would contain the output predictions, are prunned. This results in a smaller, optimized network.

The code on how to freeze and save keras models in previous versions of tensorflow is also available. Back then, the freeze_graph tool (```/tensorflow/python/tools/freeze_graph.py```) was used to convert the variables into constants. This functionality is now handled by ```graph_util.convert_variables_to_constants```
