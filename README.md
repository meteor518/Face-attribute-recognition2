# Face-attribute-recognition2

基于Face-attribute-recognition训练的模型，提取模型的倒数第二层作为特征，另外结合dlib的68点信息，人为提取脸型的特征，如脸型轮廓的曲率、相邻点的斜率、角度等。结合所有的特征作为新的特征，分别利用BP、SVM再训练，进行脸型的分类测试，以希望提高识别准确率。

* FaceDect.py：
代码中用到的dlib预模型可以从[网站](https://zh.osdn.net/projects/sfnet_dclib/downloads/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/)下载。
* get_dlib_fetures.py：利用dlib提取脸型特征
* get_model_fetures.py：利用[Face-attribute-recognition]()训练的模型，提取倒数第二层特征。
* main_bp.py：利用BP网络训练
```shell
python main_bp.py -fdt ./train/features_dlib68_train.npy -fmt ./train/features_model_train.npy -tl ./train/label.npy
```
* main_svm.py：利用SVM进行分类预测
```shell
python main_svm.py -fdt ./train/features_dlib68_train.npy -fmt ./train/features_model_train.npy -tl ./train/label.npy
```
