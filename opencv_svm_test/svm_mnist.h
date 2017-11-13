#ifndef SVM_MNIST_H
#define SVM_MNIST_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <fstream>

using namespace cv;
using namespace std;

void mnistAccuracyTest();
void randomImageTest();
int  reverseInt(int i);
void mnistTrain();


#endif // SVM_MNIST_H
