import tensorflow as tf
import numpy as np
def funcy1():
    SEED = 23455

    # 伪随机数是用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列
    rdm = np.random.RandomState(seed=SEED)  # 生成[0,1)之间的随机数
    x = rdm.rand(32, 2)  # 生成32行两列的输入特征，包含了32组随机数 x1 x2
    y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
    x = tf.cast(x, dtype=tf.float32)

    w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))  # 初始化为两行一列

    epoch = 15000
    lr = 0.002

    for epoch in range(epoch):
        with tf.GradientTape() as tape:
            y = tf.matmul(x, w1)
            loss_mse = tf.reduce_mean(tf.square(y_ - y))

        grads = tape.gradient(loss_mse, w1) # 损失函数对w1求偏导
        w1.assign_sub(lr * grads)  # 更新参数w1

        if epoch % 500 == 0:
            print("After %d training steps,w1 is " % (epoch))
            print(w1.numpy(), "\n")
    print("Final w1 is: ", w1.numpy())

# tf.convert_to_tensor()#转换数据到tensor

def myUnique(npArr, hang):  # 去部分重复
    UniqueFlagList = []
    for i in range(hang):
        for j in range(1, hang):
            if all(npArr[i] != npArr[(j + i) % hang]):
                UniqueFlagList.append(True)
            else:
                UniqueFlagList.append(False)
    if UniqueFlagList.count(True) > int((hang)):
        return True
    return False


def random_pro(varsNum, hang):
    while 1:
        rdm = np.random.RandomState()  # 生成[0,1)之间的随机数
        w1 = rdm.rand(hang, varsNum)
        for list in w1:
            for i in range(len(list)):
                if list[i] > 0.5:
                    list[i] = 1
                else:
                    list[i] = 0
        return w1

def removedep(npArr):
    resList = []
    tmpList = npArr.tolist()
    for ele1 in tmpList:
        if ele1 not in resList:
            resList.append(ele1)
    return resList


if __name__ == '__main__':

    res = np.full((4, 10), 0)
    w1 = random_pro(4, 10)
    w1 = removedep(w1)
    w1 = np.array(w1)
    x = tf.cast(w1, dtype = tf.int8)
    shape = x.get_shape().as_list()
    print("代数最高阶：", shape[1])
    print("行数：", shape[0])
    print(x)
    # print(w2)