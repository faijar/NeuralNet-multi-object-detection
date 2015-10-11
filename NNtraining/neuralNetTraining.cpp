// TrainNetwork.cpp : Defines the entry point for the console application.

#include "opencv2/opencv.hpp"    // opencv general include file
#include "opencv2/ml/ml.hpp"          // opencv machine learning include file
#include <stdio.h>
#include <fstream>
#include <string>
using namespace std;
using namespace cv;
/******************************************************************************/

#define TRAINING_SAMPLES 2468   //625       //Number of samples in training dataset
#define ATTRIBUTES 875  //Number of pixels per sample.16X16
#define TEST_SAMPLES 500         //424      //Number of samples in test dataset
#define CLASSES 6                  //Number of distinct labels.

float jumIndex = 0;
float GlobalError = 0;
float GlobalErrorTrain = 0;
int cls[8068];
int cls3[8200];
int cls2[8000];
int *c;
const char* label[6] = {"button", "text", "image", "datepicker", "dropdown", "tab"};
/********************************************************************************
This function will read the csv files(training and test dataset) and convert them
into two matrices. classes matrix have 10 columns, one column for each class label. If the label of nth row in data matrix
is, lets say 5 then the value of classes[n][5] = 1.
********************************************************************************/
void read_dataset(const char* filename, cv::Mat &data, cv::Mat &classes, cv::Mat &classes2, int clss[], int total_samples)
{

	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename, "r");

	//read each row of the csv file
	for (int baris = 0; baris < total_samples; baris++)
	{
		//for each attribute in the row
		for (int kol = 0; kol <= ATTRIBUTES; kol++)
		{
			//if its the pixel value.
			if (kol < ATTRIBUTES){

				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(baris, kol) = pixelvalue;

			}//if its the label
			else if (kol == ATTRIBUTES){
				//make the value of label column in that row as 1.
				fscanf(inputfile, "%i", &label);
				classes.at<float>(baris, label) = 1.0;
				
			}
		}
	}

	fclose(inputfile);

}

int * getClass(const char *filename, int lab[], int total_samples){
	int label;
	//int lab[1200];
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename, "r");

	//read each row of the csv file
	for (int baris = 0; baris < total_samples; baris++)
	{
		//for each attribute in the row
		for (int kol = 0; kol <= ATTRIBUTES; kol++)
		{
			//if its the pixel value.
			if (kol < ATTRIBUTES){
				fscanf(inputfile, "%f,", &pixelvalue);
			}//if its the label
			else if (kol == ATTRIBUTES){
				//make the value of label column in that row as 1.
				fscanf(inputfile, "%i", &label);
				lab[baris] = label;
				printf("label %i \n", lab[baris]);
			}
		}
	}

	fclose(inputfile);
	return lab;
}

/******************************************************************************/

int main(int argc, char** argv)
{
	const char* training = "training.txt";
	const char* testing = "testing.txt"; 
	int inputLayerSize = ATTRIBUTES;
    	int outputLayerSize = CLASSES;
	//matrix to hold the training sample
	cv::Mat training_set(TRAINING_SAMPLES, ATTRIBUTES, CV_32F);
	//matrix to hold the labels of each taining sample
	cv::Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);
	cv::Mat training_set_class(TRAINING_SAMPLES, CLASSES, CV_32F);
	//matric to hold the test samples
	cv::Mat test_set(TEST_SAMPLES, ATTRIBUTES, CV_32F);
	//matrix to hold the test labels.
	cv::Mat test_set_classifications(TEST_SAMPLES, CLASSES, CV_32F);
	cv::Mat test_set_class(TEST_SAMPLES, CLASSES, CV_32F);

	//
	cv::Mat classificationResult(1, CLASSES, CV_32F);
	cv::Mat classificationResultTrain(1, CLASSES, CV_32F);
	//load the training and test data sets.
	read_dataset(training, training_set, training_set_classifications, training_set_class, cls2, TRAINING_SAMPLES);
	read_dataset(testing, test_set, test_set_classifications, test_set_class,cls, TEST_SAMPLES);
	c = getClass(testing,cls3, TEST_SAMPLES);
	//cout << training_set.rows();
	printf("training set col %i \n", training_set.cols );
	printf("training set classification col %i \n", training_set_classifications.cols );
	// define the structure for the neural network (MLP)
	// The neural network has 3 layers.
	// - one input node per attribute in a sample so 256 input nodes
	// - 16 hidden nodes
	// - 10 output node, one for each class.
	int layerss[] = {ATTRIBUTES, CLASSES};
	vector<int> layerSizes(ATTRIBUTES, CLASSES);
        Ptr<ml::ANN_MLP> nnetwork = ml::ANN_MLP::create();
	
	//nnetwork->setTrainMethod()
	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES;//input layer
	layers.at<int>(1, 0) = 2400 ;//hidden layer
	layers.at<int>(2, 0) = CLASSES;//output layer
	nnetwork->setLayerSizes( layers );
	
	nnetwork->setActivationFunction( cv::ml::ANN_MLP::SIGMOID_SYM, 0.3, 0.5 );
	nnetwork->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.3, 0.2);

	//create the neural network.
	//for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
	
	/*
	ANN_MLP::Params params;
	params.activateFunc = ml::ANN_MLP::SIGMOID_SYM;
	params.layerSizes = layerSizes;
	params.fparam1 = 0.6;
	params.fparam2 = 1;
	params.termCrit = TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1000, 0.000001);
	params.trainMethod = ml::ANN_MLP::Params::BACKPROP;
	params.bpDWScale = 0.1;
	params.bpMomentScale = 0.1;
	params.rpDW0 = 0.1; 
	params.rpDWPlus = 1.2; 
	params.rpDWMinus = 0.5;
	params.rpDWMin = FLT_EPSILON; 
	params.rpDWMax = 50.;

	//CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM, 0.1, 0.2);
	
	CvANN_MLP_TrainParams params(

		// terminate the training after either 1000
		// iterations or a very small change in the
		// network wieghts below the specified value
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000000, 0.000001),
		// use backpropogation for training
		CvANN_MLP_TrainParams::BACKPROP,
		// co-efficents for backpropogation training
		// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
		0.1,
		0.2);
	*/
	// train the neural network (using training data)

	printf("\nUsing training dataset\n");
	//int iterations = nnetwork.train(training_set, training_set_classifications, cv::Mat(), cv::Mat(), params);
	//nnetwork->setParams(params);

	int iterations = nnetwork->train(training_set,ml::ROW_SAMPLE,training_set_classifications);


	printf("Training iterations: %i\n\n", iterations);

	// Save the model generated into an xml file.
	FileStorage fs("model.xml", FileStorage::WRITE);

	nnetwork->write(fs);
	fs.release();
	
	//CvFileStorage* storage = cvOpenFileStorage("model.xml", 0, CV_STORAGE_WRITE);
	//nnetwork.write(storage, "DigitOCR");
	//cvReleaseFileStorage(&storage);

	// Test the generated model with the test samples.
	cv::Mat test_sample;
	cv::Mat test_class;
	cv::Mat train_sample;
	//count of correct classifications
	int correct_class = 0;
	//count of wrong classifications
	int wrong_class = 0;

	//classification matrix gives the count of classes to which the samples were classified.
	int classification_matrix[CLASSES][CLASSES] = { {} };
	// for each sample in the test set.
	for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {

		// extract the sample

		test_sample = test_set.row(tsample);
		test_class = test_set_class.row(tsample);
		//try to predict its class

		nnetwork->predict(test_sample, classificationResult);
		/*The classification result matrix holds weightage  of each class.
		we take the class with the highest weightage as the resultant class */
		
		// find the class with maximum weightage.
		int maxIndex = 0;
		float value = 0.0f;
		float maxValue = classificationResult.at<float>(0, 0);
		for (int index = 1; index<CLASSES; index++)
		{
			value = classificationResult.at<float>(0, index);
			if (value>maxValue)
			{
				maxValue = value;
				maxIndex = index;
			}
		}
		jumIndex = jumIndex + maxIndex;
		printf("Testing Sample %i ->  class index %i\n", tsample, maxIndex);
		printf("row %i \n", cls3[tsample]);
		//printf("col %f\n", test_class.row(1));
		//printf("class %f\n", test_set_classifications.);

		//Now compare the predicted class to the actural class. if the prediction is correct then\
		            //test_set_classifications[tsample][ maxIndex] should be 1.
		//if the classification is wrong, note that.
		if (test_set_classifications.at<float>(tsample, maxIndex) != 1.0f)
		{
			// if they differ more than floating point error => wrong class
			//printf("coba error %i\n", test_set_classifications.at<float>(tsample, maxIndex));
			wrong_class++;

			//find the actual label 'class_index'
			for (int class_index = 0; class_index<CLASSES; class_index++)
			{
				if (test_set_classifications.at<float>(tsample, class_index) == 1.0f)
				{

					printf("aneh %i \n", cls3[class_index]);
					int Error = cls3[class_index] - maxIndex;
					printf("error %i \n",Error);
					int SE = Error*Error;
					GlobalError = GlobalError + SE;
					
					classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
					break;
				}
			}

		}
		else {

			// otherwise correct

			correct_class++;
			classification_matrix[maxIndex][maxIndex]++;
		}
	}
	float MSE = GlobalError / TEST_SAMPLES;
	float RMSE = sqrt(MSE);
	float MSETraining = GlobalErrorTrain / TRAINING_SAMPLES;
	float RMSETraining = sqrtf(MSETraining);
	printf("\nResults on the testing dataset\n"
		"\tCorrect classification: %d (%g%%)\n"
		"\tWrong classifications: %d (%g%%)\n",
		correct_class, (double)correct_class * 100 / TEST_SAMPLES,
		wrong_class, (double)wrong_class * 100 / TEST_SAMPLES);
	float test = sqrt((TEST_SAMPLES - correct_class) / TEST_SAMPLES);
	printf("RMSE %f \n", RMSE);
	printf("test %f \n", test);
	printf("RMSE training %f \n", RMSETraining);
	cout << "   ";
	/*
	for (int i = 0; i < CLASSES; i++)
	{
		cout << i << "\t";
	}
	cout << "\n";
	for (int row = 0; row<CLASSES; row++)
	{
		cout << row << "  ";
		for (int col = 0; col<CLASSES; col++)
		{
			cout << classification_matrix[row][col] << "\t";
		}
		cout << "\n";
	}*/
	std::getchar();
	return 0;

}
