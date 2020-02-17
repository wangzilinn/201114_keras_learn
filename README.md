# Keras Learn
这个工程既是Keras的学习工程, 准备在其中实现各种类型的神经网络, 也是为[200116_MCU_NN_learn](https://github.com/wangzilinn/200116_MCU_NN_learn)(private)库生成.h5文件和代码的预备工程

目前, 实现的神经网络有:

1. `regressor.py` 一个单神经元的回归神经网络
2. `classifier.py` 一个基于MNIST的分类器
3. `RNN_classifier.py` 一个RNN分类器
4. ...

此外, 还编写了几个辅助的python文件,:

1. `test.py` 用于检验生成的伸进网络, 并进行可视化
2. `code_generator.py` 用于生成单片机中作为神经网络输入参数的static数组
