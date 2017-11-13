#include "svm_mnist.h"

const char* keys =
{
       "{c |type  |  3  | 1. train 2. test_Accuacy 3.test_Image }"
};
int  main(int argc,char* argv[])
{

     CommandLineParser parser(argc, argv, keys);
     int type = parser.get<int>("type");
     cout<<"type:"<<type<<endl;
     switch (type) {
     case 1:
     cout << "mnistTrain"<<endl;
     mnistTrain();
         break;
     case 2:
     cout << "mnistAccuracyTest"<<endl;
     mnistAccuracyTest();
         break;
     case 3:
     cout << "randomImageTest"<<endl;
     randomImageTest();
         break;
     default:
         break;
     }
    waitKey(0);
    return 0;
}


