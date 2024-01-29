//void load_and_predict()
//{
//	//!!!!!!!!!!!!!!!!! remember to change the LocalFileDatasetLoader to ommit the class
//
//	phd::svm::libSvmImplementation svm{ R"(D:\PHD\ESA\1\2019-11-28-00_01_48.726__GridSearchNoFS1_0fold_1_svmModel.xml)" };
//
//	auto tr = filesystem::Path(R"(D:\PHD\ESA\1\train.csv)");
//	auto test = filesystem::Path(R"(D:\PHD\ESA\1\test.csv)");
//
//	genetic::LocalFileDatasetLoader loader{ tr,test,test };
//
//
//	auto testset = loader.getTestSet();
//
//	auto samples = testset.getSamples();
//
//	std::ofstream output{ R"(D:\PHD\ESA\1\2019-11-28-00_01_48.726__GridSearchNoFS1_0fold_1_svmModel_results.txt)", std::fstream::out };
//
//	//output.open("D:\PHD\ESA\1\MASVM_512\2019-12-05-11_00_12.195__MASVM1_0fold_1_svmModel_results.txt");
//	if (output)
//	{
//		std::cout << "File correct";
//
//		for (auto& sample : samples)
//		{
//			auto result = svm.classify(sample);
//			output << result << "\n";
//		}
//
//		output.close();
//	}
//}


////inline void testLoadedSvm()
////{
////    auto svm = loadSvm(R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\ALGA24\2017-11-22-11_03_38.614___0fold_1_svmModel.xml)");
////
////
////    genetic::LocalFileDatasetLoader loader(R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\train1.csv)",
////                                           R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\train1.csv)",
////                                           R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\test1.csv)");
////
////
////    svmComponents::BaseSvmChromosome chromosome;
////    chromosome.updateClassifier(svm);
////
////    svmComponents::SvmAccuracyMetric acc;
////    auto result = acc.calculateMetric(chromosome, loader.getValidationSet());
////
////    std::cout << "Fitness: " << result.m_fitness << "\n ConfusionMatrix: " << result.m_confusionMatrix;
////}
//
////inline void dumpScalling()
////{
////    for (int i = 1; i < 6; ++i)
////    {
////        genetic::LocalFileDatasetLoader loader("C:\\Users\\wdudzik\\Desktop\\Experiments\\SuperPixelsFolds\\train" +std::to_string(i) + ".csv",
////                                               R"(C:\Users\wdudzik\Desktop\Experiments\SuperPixelsFolds\test1.csv)",
////                                               R"(C:\Users\wdudzik\Desktop\Experiments\SuperPixelsFolds\test1.csv)");
////
////        loader.getTraningSet();
////    }
////}
//
////void testSvmSave()
////{
////    using namespace cv;
////    using namespace cv::ml;
////
////    // Data for visual representation
////    int width = 512, height = 512;
////    Mat image = Mat::zeros(height, width, CV_8UC3);
////
////    // Set up training data
////    int labels[4] = { 1, -1, -1, -1 };
////    Mat labelsMat(4, 1, CV_32SC1, labels);
////
////    float trainingData[4][2] = { { 501, 10 },{ 255, 10 },{ 501, 255 },{ 10, 501 } };
////    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
////
////    // Set up SVM's parameters
////    Ptr<ml::SVM> svm = ml::SVM::create();
////    svm->setType(ml::SVM::C_SVC);
////    svm->setKernel(SVM::INTER); // Algorithm::load() works well with SVM::LINEAR
////    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
////
////    // Train the SVM
////    svm->train(ml::TrainData::create(trainingDataMat, ml::ROW_SAMPLE, labelsMat));
////
////    // Save and load SVM
////    svm->save("ex_svm.xml");
////    svm = cv::Algorithm::load<ml::SVM>("ex_svm.xml"); // something is wrong
////
////
////    Vec3b green(0, 255, 0), blue(255, 0, 0);
////    // Show the decision regions given by the SVM
////    for (int i = 0; i < image.rows; ++i)
////        for (int j = 0; j < image.cols; ++j)
////        {
////            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
////            float response = svm->predict(sampleMat);
////
////            if (response == 1)
////                image.at<Vec3b>(i, j) = green;
////            else if (response == -1)
////                image.at<Vec3b>(i, j) = blue;
////        }
////
////    // Show the training data
////    int thickness = -1;
////    int lineType = 8;
////    circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
////    circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
////    circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
////    circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
////
////    // Show support vectors
////    thickness = 2;
////    lineType = 8;
////
////    Mat sv = svm->getSupportVectors();
////
////    for (int i = 0; i < sv.rows; ++i)
////    {
////        const float* v = sv.ptr<float>(i);
////        circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
////    }
////
////    imwrite("result.png", image);        // save the image
////
////    imshow("SVM Simple Example", image); // show it to the user
////    waitKey(0);
////}
//
//inline void testAuc()
//{
//    phd::svm::OpenCvSvm svm(filesystem::Path(R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\2018-02-14-12_52_02.240___7fold_1_svmModel.xml)"));
//
//    auto SvmPtr = std::make_unique<phd::svm::OpenCvSvm>(svm);
//
//    genetic::LocalFileDatasetLoader loader(R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\train1.csv)",
//                                           R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\train1.csv)",
//                                           R"(C:\Users\wdudzik\Desktop\Experiments14.02.2018\test\test1.csv)");
//
//
//    svmComponents::BaseSvmChromosome chromosome;
//    chromosome.updateClassifier(std::move(SvmPtr));
//
//    svmComponents::SvmAucMetric aucMetric;
//
//    std::vector<bool> feat = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1 };
//
//
//    //111111111111110110111011
//    svmComponents::SvmFeatureSetChromosome features(std::move(feat));
//    const auto convertedSet = features.convertChromosome(loader.getValidationSet());
//
//
//    aucMetric.calculateMetric(chromosome, convertedSet);
//
//    //    std::cout << "Fitness: " << result.m_fitness << "\n ConfusionMatrix: " << result.m_confusionMatrix;
//}


//Testowanie wizualizacji
//inline int normalizeToImageHeight(double value)
//{
//	//auto height = static_cast<int>(value * (500 - 1));
//	auto height = std::round(value * (500));

//	if (height < 0 || height > 500)
//		return 0;

//	return static_cast<int>(height);
//}

/*dataset::Dataset<std::vector<float>, float> d,d1,d2;
for (int i = 0; i < 500; ++i)
{
	d.addSample(std::move(std::vector<float>{ static_cast<float>(i),static_cast<float>(i) }), 1.0);
	d1.addSample(std::move(std::vector<float>{ static_cast<float>(i), static_cast<float>(i) }), 1.0);
	d2.addSample(std::move(std::vector<float>{ static_cast<float>(i), static_cast<float>(i) }), 1.0);
}

auto normalizer = svmComponents::DataNormalization(2);

normalizer.normalize(d, d1, d2);

int i = 0;
for(const auto sample : d.getSamples())
{
	std::cout << normalizeToImageHeight(sample[0]) << "\n";
	if(normalizeToImageHeight(sample[0]) != i)
	{
		std::cout << "Error\n";
		++i;
	}
	++i;
}*/