# k-近邻算法
import os

import numpy
import operator
import matplotlib
import matplotlib.pyplot as plt


def create_data_set():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    :param inX:用于分类的输入向量
    :param dataSet:输入的训练样本集
    :param labels:标签向量
    :param k:表示用于选择最近邻居的数目
    :return:
    """
    # 返回各个维度的维数
    dataSetSize = dataSet.shape[0]
    # 生成与训练样本同纬数的数组
    inXAry = numpy.tile(inX, (dataSetSize, 1))
    # 使用欧氏距离计算点与点之间的差值
    diffMat = inXAry - dataSet
    # 计算平方
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    # 开平方
    distances = sqDistances ** 0.5

    # 函数argsort() 将distances中的元素从小到大排列，提取其对应的index(索引)
    sortedDistIndicies = distances.argsort()

    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 累加同类别
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    returnMat = numpy.zeros((number_of_lines, 3))
    classLabelVector = []
    index = 0

    for line in array_of_lines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    # 每列的最小值
    minVals = dataSet.min(0)
    # 每列的最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = numpy.zeros(numpy.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is:%f" % (errorCount / float(numTestVecs)))


def img2vector(filename):
    returnVect = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)

    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        if classifierResult != classNumStr:
            print("the classifier came back with:%d, the real answer is:%d fileName:%s" % (classifierResult, classNumStr, fileNameStr))
            errorCount += 1
    print("the total number of errors is : %d" % errorCount)
    print("the total error rate is : %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    # group, labels = create_data_set()
    # ret = classify0([0.5, 0.5], group, labels, 3)
    # print(ret)

    # mat, vector = file2matrix("datingTestSet2.txt")
    # # print(mat)
    # # print(vector)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.scatter(mat[:, 1], mat[:, 2])
    # ax.scatter(mat[:, 1], mat[:, 2], 10.0 * numpy.array(vector), 10.0 * numpy.array(vector))
    # plt.show()

    # normDataSet, ranges, minVals = autoNorm(mat)

    # datingClassTest()

    handwritingClassTest()
