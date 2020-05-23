import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))  # 将W的随机初始值设定为5，设置为可训练
lr = 0.2  # 当学习率较小时，学习速度慢；若参数过大，则结果在最优值两边跳动
epoch = 40

for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环100次迭代。
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)  # 对w + 1 求平方方
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导，对loss函数的w进行求梯度

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f" % (epoch, w.numpy(), loss))

# lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数w
