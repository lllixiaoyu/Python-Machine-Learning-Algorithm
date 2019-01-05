# coding:UTF-8

import numpy as np
import svm

def load_data_libsvm(data_file):
    '''导入训练数据
    input:  data_file(string):训练数据所在文件
    output: data(mat):训练样本的特征
            label(mat):训练样本的标签
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        
        # 提取得出label
        label.append(float(lines[0]))
        # 提取出特征，并将其放入到矩阵中
        index = 0
        tmp = []
        for i in xrange(1, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data), np.mat(label).T

def cal_accuracy(svm, test_x, test_y):
	'''计算预测的准确性
	input:svm : SVM 模型
		test_x(mat):测试的特征
		test_y(mat):测试的标签
	output:accuacy(float):预测的准确性
	'''
	n_samples = np.shape(test_x)[0]
	correct = 0.0
	for i in range(n_samples):
		#对每一个样本得到预测值
		predict = svm_predict(svm, test_x[i, :])
		#判断每一个样本值的预测值与真实值是否一样
		if np.sign(predict) == np.sign(test_y[1]):
			correct += 1
	accuracy = correct / n_samples
	return  accuracy

def svm_predict(svm, test_sample_x):
	'''利用SVM模型对每一个样本进行预测
	input:svm:SVM模型
		test_sample_x(mat):样本
	output:predict(float)：对样本的预测
	'''
	#计算核函数矩阵
	kernel_value = svm.cal_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
	#计算预测值
	predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
	return predict


def save_svm_model(svm_model, model_file):
	'''
	保存svm模型
	input:param svm_model: SVM模型
	:param model_file: SVM模型需要保存的文件
	'''
	with open(model_file, 'w') as f :
		pickle.dump(svm.model, f)

if __name__ == "__main__":
    # 1、导入训练数据
    print "------------ 1、load data --------------"
    dataSet, labels = load_data_libsvm("heart_scale")
    # 2、训练SVM模型
    print "------------ 2、training ---------------"
    C = 0.6
    toler = 0.001
    maxIter = 500
    svm_model = svm.SVM_training(dataSet, labels, C, toler, maxIter)
    # 3、计算训练的准确性
    print "------------ 3、cal accuracy --------------"
    accuracy = svm.cal_accuracy(svm_model, dataSet, labels)  
    print "The training accuracy is: %.3f%%" % (accuracy * 100)
    # 4、保存最终的SVM模型
    print "------------ 4、save model ----------------"
    svm.save_svm_model(svm_model, "model_file")
