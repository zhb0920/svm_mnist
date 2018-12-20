
#include "svm_mnist.h"

void mnistTrain()
{
    //读取训练样本集
    ifstream if_trainImags("train-images.idx3-ubyte", ios::binary);
    //读取失败
    if (true == if_trainImags.fail())
    {
        cout << "Please check the path of file train-images-idx3-ubyte" << endl;
        return ;
    }
    int magic_num, trainImgsNum, nrows, ncols;
    //读取magic number
    if_trainImags.read((char*)&magic_num, sizeof(magic_num));
    magic_num = reverseInt(magic_num);
    cout << "训练图像数据库train-images-idx3-ubyte的magic number为：" << magic_num << endl;
    //读取训练图像总数
    if_trainImags.read((char*)&trainImgsNum, sizeof(trainImgsNum));
    trainImgsNum = reverseInt(trainImgsNum);
    cout << "训练图像数据库train-images-idx3-ubyte的图像总数为：" << trainImgsNum << endl;
    //读取图像的行大小
    if_trainImags.read((char*)&nrows, sizeof(nrows));
    nrows = reverseInt(nrows);
    cout << "训练图像数据库train-images-idx3-ubyte的图像维度row为：" << nrows << endl;
    //读取图像的列大小
    if_trainImags.read((char*)&ncols, sizeof(ncols));
    ncols = reverseInt(ncols);
    cout << "训练图像数据库train-images-idx3-ubyte的图像维度col为：" << ncols << endl;

    //读取训练图像
    int imgVectorLen = nrows * ncols;
    Mat trainFeatures = Mat::zeros(trainImgsNum, imgVectorLen, CV_32FC1);
    Mat temp = Mat::zeros(nrows, ncols, CV_8UC1);
    for (int i = 0; i < trainImgsNum; i++)
    {
        if_trainImags.read((char*)temp.data, imgVectorLen);
        Mat tempFloat;
        //由于SVM需要的训练数据格式是CV_32FC1，在这里进行转换
        temp.convertTo(tempFloat, CV_32FC1);
        memcpy(trainFeatures.data+i*imgVectorLen *sizeof(float), tempFloat.data, imgVectorLen * sizeof(float));
    }
    //归一化
    trainFeatures = trainFeatures / 255;
    //读取训练图像对应的分类标签
    ifstream if_trainLabels("train-labels.idx1-ubyte", ios::binary);
    //读取失败
    if (true == if_trainLabels.fail())
    {
        cout << "Please check the path of file train-labels-idx1-ubyte" << endl;
        return ;
    }
    int magic_num_2, trainLblsNum;
    //读取magic number
    if_trainLabels.read((char*)&magic_num_2, sizeof(magic_num_2));
    magic_num_2 = reverseInt(magic_num_2);
    cout << "训练图像标签数据库train-labels-idx1-ubyte的magic number为：" << magic_num_2 << endl;
    //读取训练图像的分类标签的数量
    if_trainLabels.read((char*)&trainLblsNum, sizeof(trainLblsNum));
    trainLblsNum = reverseInt(trainLblsNum);
    cout << "训练图像标签数据库train-labels-idx1-ubyte的标签总数为：" << trainLblsNum << endl;

    //由于SVM需要输入的标签类型是CV_32SC1，在这里进行转换
    Mat trainLabels = Mat::zeros(trainLblsNum, 1, CV_32SC1);
    Mat readLabels  = Mat::zeros(trainLblsNum, 1, CV_8UC1);
    if_trainLabels.read((char*)readLabels.data, trainLblsNum*sizeof(char));
    readLabels.convertTo(trainLabels, CV_32SC1);
    // 训练SVM分类器

    CvSVM svm;
    CvSVMParams param;
    CvTermCriteria criteria;
    criteria= cvTermCriteria(CV_TERMCRIT_EPS, 200, FLT_EPSILON);
    param= CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.01, 1.0, 10.0, 0.5, 0.1, NULL, criteria);

    svm.train(trainFeatures, trainLabels, Mat(), Mat(), param);
    cout << "训练结束，正写入xml:" << endl;
    svm.save( "mnist.xml" );
    return ;

}


void mnistAccuracyTest()
{
//读取测试样本集
ifstream if_testImags("t10k-images.idx3-ubyte", ios::binary);
//读取失败
if (true == if_testImags.fail())
{
    cout << "Please check the path of file t10k-images-idx3-ubyte" << endl;
    return ;
}
int magic_num, testImgsNum, nrows, ncols;
//读取magic number
if_testImags.read((char*)&magic_num, sizeof(magic_num));
magic_num = reverseInt(magic_num);
cout << "测试图像数据库t10k-images-idx3-ubyte的magic number为：" << magic_num << endl;
//读取测试图像总数
if_testImags.read((char*)&testImgsNum, sizeof(testImgsNum));
testImgsNum = reverseInt(testImgsNum);
cout << "测试图像数据库t10k-images-idx3-ubyte的图像总数为：" << testImgsNum << endl;
//读取图像的行大小
if_testImags.read((char*)&nrows, sizeof(nrows));
nrows = reverseInt(nrows);
cout << "测试图像数据库t10k-images-idx3-ubyte的图像维度row为：" << nrows << endl;
//读取图像的列大小
if_testImags.read((char*)&ncols, sizeof(ncols));
ncols = reverseInt(ncols);
cout << "测试图像数据库t10k-images-idx3-ubyte的图像维度col为：" << ncols << endl;

//读取测试图像
int imgVectorLen = nrows * ncols;
Mat testFeatures = Mat::zeros(testImgsNum, imgVectorLen, CV_32FC1);
Mat temp = Mat::zeros(nrows, ncols, CV_8UC1);
for (int i = 0; i < testImgsNum; i++)
{
    if_testImags.read((char*)temp.data, imgVectorLen);
    Mat tempFloat;
    //由于SVM需要的测试数据格式是CV_32FC1，在这里进行转换
    temp.convertTo(tempFloat, CV_32FC1);
    memcpy(testFeatures.data + i*imgVectorLen * sizeof(float), tempFloat.data, imgVectorLen * sizeof(float));
}
//归一化
testFeatures = testFeatures / 255;
//读取测试图像对应的分类标签
ifstream if_testLabels("t10k-labels.idx1-ubyte", ios::binary);
//读取失败
if (true == if_testLabels.fail())
{
    cout << "Please check the path of file t10k-labels-idx1-ubyte" << endl;
    return ;
}
int magic_num_2, testLblsNum;
//读取magic number
if_testLabels.read((char*)&magic_num_2, sizeof(magic_num_2));
magic_num_2 = reverseInt(magic_num_2);
cout << "测试图像标签数据库t10k-labels-idx1-ubyte的magic number为：" << magic_num_2 << endl;
//读取测试图像的分类标签的数量
if_testLabels.read((char*)&testLblsNum, sizeof(testLblsNum));
testLblsNum = reverseInt(testLblsNum);
cout << "测试图像标签数据库t10k-labels-idx1-ubyte的标签总数为：" << testLblsNum << endl;

//由于SVM需要输入的标签类型是CV_32SC1，在这里进行转换
Mat testLabels = Mat::zeros(testLblsNum, 1, CV_32SC1);
Mat readLabels = Mat::zeros(testLblsNum, 1, CV_8UC1);
if_testLabels.read((char*)readLabels.data, testLblsNum * sizeof(char));
readLabels.convertTo(testLabels, CV_32SC1);

//载入训练好的SVM模型
CvSVM svm;
svm.load("mnist.xml");

int sum = 0;
//对每一个测试图像进行SVM分类预测
for (int i = 0; i < testLblsNum; i++)
{
    Mat predict_mat = Mat::zeros(1, imgVectorLen, CV_32FC1);
    memcpy(predict_mat.data, testFeatures.data + i*imgVectorLen * sizeof(float), imgVectorLen * sizeof(float));
    //预测
    float predict_label = svm.predict(predict_mat);
    //真实的样本标签
    float truth_label = testLabels.at<int>(i);
    //比较判定是否预测正确
    if ((int)predict_label == (int)truth_label)
    {
        sum++;
    }
}

cout << "预测准确率为："<<(double)sum / (double)testLblsNum << endl;
}


void randomImageTest()
{
    //读取测试样本集
    ifstream if_testImags("t10k-images.idx3-ubyte", ios::binary);
    //读取失败
    if (true == if_testImags.fail())
    {
        cout << "Please check the path of file t10k-images-idx3-ubyte" << endl;
        return ;
    }
    int magic_num, testImgsNum, nrows, ncols;
    //读取magic number
    if_testImags.read((char*)&magic_num, sizeof(magic_num));
    magic_num = reverseInt(magic_num);
    cout << "测试图像数据库t10k-images-idx3-ubyte的magic number为：" << magic_num << endl;
    //读取测试图像总数
    if_testImags.read((char*)&testImgsNum, sizeof(testImgsNum));
    testImgsNum = reverseInt(testImgsNum);
    cout << "测试图像数据库t10k-images-idx3-ubyte的图像总数为：" << testImgsNum << endl;
    //读取图像的行大小
    if_testImags.read((char*)&nrows, sizeof(nrows));
    nrows = reverseInt(nrows);
    cout << "测试图像数据库t10k-images-idx3-ubyte的图像维度row为：" << nrows << endl;
    //读取图像的列大小
    if_testImags.read((char*)&ncols, sizeof(ncols));
    ncols = reverseInt(ncols);
    cout << "测试图像数据库t10k-images-idx3-ubyte的图像维度col为：" << ncols << endl;

    //读取测试图像
    int imgVectorLen = nrows * ncols;
    Mat testFeatures = Mat::zeros(testImgsNum, imgVectorLen, CV_32FC1);
    Mat temp = Mat::zeros(nrows, ncols, CV_8UC1);
    for (int i = 0; i < testImgsNum; i++)
    {
        if_testImags.read((char*)temp.data, imgVectorLen);
        Mat tempFloat;
        //由于SVM需要的测试数据格式是CV_32FC1，在这里进行转换
        temp.convertTo(tempFloat, CV_32FC1);
        memcpy(testFeatures.data + i*imgVectorLen * sizeof(float), tempFloat.data, imgVectorLen * sizeof(float));
    }
    //归一化
    testFeatures = testFeatures / 255;
    //读取测试图像对应的分类标签
    ifstream if_testLabels("t10k-labels.idx1-ubyte", ios::binary);
    //读取失败
    if (true == if_testLabels.fail())
    {
        cout << "Please check the path of file t10k-labels-idx1-ubyte" << endl;
        return;
    }
    int magic_num_2, testLblsNum;
    //读取magic number
    if_testLabels.read((char*)&magic_num_2, sizeof(magic_num_2));
    magic_num_2 = reverseInt(magic_num_2);
    cout << "测试图像标签数据库t10k-labels-idx1-ubyte的magic number为：" << magic_num_2 << endl;
    //读取测试图像的分类标签的数量
    if_testLabels.read((char*)&testLblsNum, sizeof(testLblsNum));
    testLblsNum = reverseInt(testLblsNum);
    cout << "测试图像标签数据库t10k-labels-idx1-ubyte的标签总数为：" << testLblsNum << endl;

    //由于SVM需要输入的标签类型是CV_32SC1，在这里进行转换
    Mat testLabels = Mat::zeros(testLblsNum, 1, CV_32SC1);
    Mat readLabels = Mat::zeros(testLblsNum, 1, CV_8UC1);
    if_testLabels.read((char*)readLabels.data, testLblsNum * sizeof(char));
    readLabels.convertTo(testLabels, CV_32SC1);

    //载入训练好的SVM模型
    CvSVM svm;
    svm.load("mnist.xml");
    //随机测试某一个图像看效果，输入为-1时退出
    while (1)
    {
        int index;
        cout << "请输入要查看的测试图像下标" << endl;
        cin >> index;
        if (-1 == index)
        {
        break;
        }
        Mat show_mat = Mat::zeros(nrows, ncols, CV_32FC1);
        Mat predict_mat = Mat::zeros(1, imgVectorLen, CV_32FC1);
        memcpy(show_mat.data, testFeatures.data + index*imgVectorLen * sizeof(float), imgVectorLen * sizeof(float));
        memcpy(predict_mat.data, testFeatures.data + index*imgVectorLen * sizeof(float), imgVectorLen * sizeof(float));
        float response = svm.predict(predict_mat);

        imshow("test", show_mat);
        cout << "标签值为" << response <<endl;
        waitKey(500);
    }

}


















//大端转小端
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
