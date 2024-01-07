

// #include <string>
// #include <vector>
// #include <fstream>

// #include <opencv2/core.hpp>
// #include <opencv2/ml.hpp>

// #include "libSvm/OpenCvSvm.h"
// #include "libSvmSigmoidTrain/SvmSigmoid.h"
// #include "SvmExceptions.h"

// #include "libPlatform/StringUtils.h"

// namespace phd { namespace svm
// {
// OpenCvSvm::OpenCvSvm()
//     : m_svm(cv::ml::SVM::create())
//     , m_sigmoidA(0)
//     , m_sigmoidB(0)
//     , m_optimalProbabilityThreshold(Nan)
// {
//     m_svm->setType(cv::ml::SVM::C_SVC);
//     m_svm->setKernel(cv::ml::SVM::LINEAR);

//     const auto terminationCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6);
//     m_svm->setTermCriteria(terminationCriteria);
// }

// OpenCvSvm::OpenCvSvm(const filesystem::Path& filepath)
//     : m_sigmoidA(0)
//     , m_sigmoidB(0)
//     , m_optimalProbabilityThreshold(Nan)
// {
//     m_svm = cv::Algorithm::load<cv::ml::SVM>(filepath.string());

//     std::ifstream svmModelFile(filepath.string());
//     std::vector<std::string> fileByLines;
//     std::copy(std::istream_iterator<std::string>(svmModelFile),
//               std::istream_iterator<std::string>(),
//               std::back_inserter(fileByLines));

//     /*@wdudzik constant values express number of line (counting from end of file) in which each value is located
//      * this is closely related to save function.
//      */
// 	loadFeatureSet(*(fileByLines.rbegin() + 2));
//     m_optimalProbabilityThreshold = std::stod(*(fileByLines.rbegin() + 5));
//     m_sigmoidB = std::stod(*(fileByLines.rbegin() + 8));
//     m_sigmoidA = std::stod(*(fileByLines.rbegin() + 11));
// }

// OpenCvSvm::OpenCvSvm(cv::Ptr<cv::ml::SVM> svm, double sigmoidA, double sigmoidB, double optimalProbabilityThreshold)
//     : m_svm(std::move(svm))
//     , m_sigmoidA(sigmoidA)
//     , m_sigmoidB(sigmoidB)
//     , m_optimalProbabilityThreshold(optimalProbabilityThreshold)
// {
// }

// KernelTypes OpenCvSvm::getKernelType() const
// {
//     return static_cast<KernelTypes>(m_svm->getKernelType());
// }

// void OpenCvSvm::setKernel(KernelTypes kernelType)
// {
//     m_svm->setKernel(static_cast<int>(kernelType));
// }

// SvmTypes OpenCvSvm::getType() const
// {
//     switch (m_svm->getType())
//     {
//     case cv::ml::SVM::Types::C_SVC:
//         return SvmTypes::CSvc;
//     case cv::ml::SVM::Types::EPS_SVR:
//         return SvmTypes::EpsSvr;
//     case cv::ml::SVM::Types::NU_SVC:
//         return SvmTypes::NuSvc;
//     case cv::ml::SVM::Types::NU_SVR:
//         return SvmTypes::NuSvr;
//     case cv::ml::SVM::Types::ONE_CLASS:
//         return SvmTypes::OneClass;
//     default:
//         throw UnknownSvmTypeException(SvmTypes::Unknown);
//     }
// }

// void OpenCvSvm::setType(SvmTypes svmType)
// {
//     switch (svmType)
//     {
//     case SvmTypes::CSvc:
//     {
//         m_svm->setType(cv::ml::SVM::C_SVC);
//         break;
//     }
//     case SvmTypes::NuSvc:
//     {
//         m_svm->setType(cv::ml::SVM::NU_SVC);
//         break;
//     }
//     case SvmTypes::OneClass:
//     {
//         m_svm->setType(cv::ml::SVM::ONE_CLASS);
//         break;
//     }
//     case SvmTypes::EpsSvr:
//     {
//         m_svm->setType(cv::ml::SVM::EPS_SVR);
//         break;
//     }
//     case SvmTypes::NuSvr:
//     {
//         m_svm->setType(cv::ml::SVM::NU_SVR);
//         break;
//     }
//     default:
//         throw UnknownSvmTypeException(svmType);
//     }
// }

// inline std::string convertToString(const std::vector<svmComponents::Feature>& featureSet, int featureNumber)
// {
// 	if (featureSet.empty())
// 	{
// 		return "";
// 	}

// 	std::vector<bool> oldRepresentation;
// 	oldRepresentation.resize(featureNumber, false);

// 	for (const auto feature : featureSet)
// 	{
// 		oldRepresentation[feature.id] = true;
// 	}

// 	std::stringstream genesAsString;
// 	std::copy(oldRepresentation.begin(), oldRepresentation.end(), std::ostream_iterator<int>(genesAsString, ","));

// 	auto str = genesAsString.str();
// 	str.pop_back();

// 	return str;
// }


// void OpenCvSvm::save(const filesystem::Path& filepath)
// {
//     m_svm->save(filepath.string());
//     std::ofstream svmModleFile(filepath.string(), std::ios_base::app | std::ios_base::out);
//     svmModleFile << "<opencv_storage>\n";
//     svmModleFile << "<sigmoidA>\n" << std::to_string(m_sigmoidA) << "\n</sigmoidA>\n";
//     svmModleFile << "<sigmoidB>\n" << std::to_string(m_sigmoidB) << "\n</sigmoidB>\n";
//     svmModleFile << "<optimalProbabilityThreshold>\n" << std::to_string(m_optimalProbabilityThreshold) << "\n</optimalProbabilityThreshold>\n";

// 	svmModleFile << "<features_set>\n" << convertToString(m_featureSet, m_numberOfFeatures) << "\n</features_set>\n";
	

//     svmModleFile << "</opencv_storage>\n";


//     svmModleFile.close();
// }

// uint32_t OpenCvSvm::getNumberOfKernelParameters(KernelTypes kernelType) const
// {
//     switch (kernelType)
//     {
//     case KernelTypes::Rbf:
//         return 2;
//     default:
//         throw UnsupportedKernelTypeException(kernelType);
//     }
// }

// uint32_t OpenCvSvm::getNumberOfSupportVectors() const
// {
//     if (getKernelType() != KernelTypes::Linear)
//     {
//         return m_svm->getSupportVectors().rows;
//     }
//     return m_svm->getUncompressedSupportVectors().rows;
// }

// cv::Mat OpenCvSvm::getSupportVectors() const
// {
//     if (getKernelType() != KernelTypes::Linear)
//     {
//         return m_svm->getSupportVectors();
//     }
//     return m_svm->getUncompressedSupportVectors();
// }

// float OpenCvSvm::classify(const gsl::span<const float> sample) const
// {
//     const auto sampleMat = cv::Mat(m_rowsPerSample, static_cast<int>(sample.size()), CV_32FC1, const_cast<float*>(sample.data()));
//     return m_svm->predict(sampleMat);
// }

// void OpenCvSvm::train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool probabilityNeeded)
// {
//     if (trainingSet.size() == 0)
//     {
//         throw EmptyTraningDataSet();
//     }
//     const auto trainingMat = createTrainingData(trainingSet);
//     m_svm->train(trainingMat);

//     if (probabilityNeeded)
//     {
//         calculateSigmoidParametrs(trainingSet);
//     }
// }

// void OpenCvSvm::train(cv::Ptr<cv::ml::TrainData> trainingSet) const
// {
//     m_svm->train(trainingSet);
// }

// cv::Ptr<cv::ml::TrainData> OpenCvSvm::createTrainingData(const dataset::Dataset<std::vector<float>, float>& trainingSet)
// {
//     const auto samples = trainingSet.getSamples();
//     const auto sampleSize = samples[0].size();

//     cv::Mat trainingDataMat = cv::Mat::zeros(
//         static_cast<int>(trainingSet.size()),
//         static_cast<int>(sampleSize),
//         CV_32FC1);

//     for (auto i = 0; i < samples.size(); ++i)
//     {
//         const auto& sampleRow = samples[i];
//         auto targetRow = trainingDataMat.row(i);
//         for (auto j = 0U; j < sampleSize; ++j)
//         {
//             targetRow.at<float>(j) = (sampleRow[j]);
//         }
//     }

//     if(m_svm->getType() == cv::ml::SVM::C_SVC)
//     {
//         auto targets = trainingSet.getLabels();
//         std::vector<int> lab;
//         std::transform(targets.begin(), targets.end(), std::back_inserter(lab), [](auto label) { return static_cast<int>(label);  });
//         const int columnsForClass = 1;
//         const cv::Mat labelsMat(static_cast<int>(targets.size()), columnsForClass, CV_32SC1, const_cast<int*>(lab.data()));

//         return cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
//     }
//     throw std::exception("Other than CSVC not implemented");
//     //if (m_svm->getType() == static_cast<int>(SvmTypes::EpsSvr))
//    /* {
//         auto targets = trainingSet.getLabels();
//         const int columnsForClass = 1;
//         const cv::Mat labelsMat(static_cast<int>(targets.size()), columnsForClass, CV_32FC1, const_cast<float*>(targets.data()));

//         return cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
//     }*/
// }

// double OpenCvSvm::sigmoidPredict(float decisionValue) const
// {
//     const auto logisticFunctionArgument = decisionValue * m_sigmoidA + m_sigmoidB;
//     if (logisticFunctionArgument >= 0)
//     {
//         return exp(-logisticFunctionArgument) / (1.0 + exp(-logisticFunctionArgument));
//     }
//     return 1.0 / (1 + exp(logisticFunctionArgument));
// }

// void OpenCvSvm::loadFeatureSet(std::string featuresDelimeted)
// {
// 	auto vectorOfBoolFeatures = platform::stringUtils::splitString(featuresDelimeted, ',');
// 	std::vector<svmComponents::Feature> featureSet;
// 	int i = 0;
// 	for (auto& feature : vectorOfBoolFeatures)
// 	{
// 		if (feature == "1")
// 		{
// 			featureSet.emplace_back(i);
// 		}
// 		++i;
// 	}
// 	m_featureSet = featureSet;
// }

// void OpenCvSvm::setFeatureSet(const std::vector<svmComponents::Feature>& features, int numberOfFeatures)
// {
// 	m_featureSet = features;
// 	m_numberOfFeatures = numberOfFeatures;
// }

// const std::vector<svmComponents::Feature>& OpenCvSvm::getFeatureSet()
// {
// 	return m_featureSet;
// }

// float OpenCvSvm::classifyHyperplaneDistance(const gsl::span<const float> sample) const
// {
//     const auto sampleMat = cv::Mat(m_rowsPerSample, static_cast<int>(sample.size()), CV_32FC1, const_cast<float*>(sample.data()));
//     cv::Mat result;
//     m_svm->predict(sampleMat, result, cv::ml::StatModel::Flags::RAW_OUTPUT);
//     return result.at<float>(0, 0);
// }

// double OpenCvSvm::classificationProbability(const gsl::span<const float> sample) const
// {
//     const auto sampleMat = cv::Mat(m_rowsPerSample, static_cast<int>(sample.size()), CV_32FC1, const_cast<float*>(sample.data()));
//     cv::Mat result;
//     m_svm->predict(sampleMat, result, cv::ml::StatModel::Flags::RAW_OUTPUT);

//     return sigmoidPredict(result.at<float>(0, 0));
// }

// void OpenCvSvm::setTerminationCriteria(const cv::TermCriteria& value)
// {
//     m_svm->setTermCriteria(value);
// }

// cv::TermCriteria OpenCvSvm::getTerminationCriteria() const
// {
//     return m_svm->getTermCriteria();
// }

// bool OpenCvSvm::isTrained() const
// {
//     return m_svm->isTrained();
// }

// bool OpenCvSvm::canGiveProbabilityOutput() const
// {
//     return !(m_sigmoidA == 0.0 && m_sigmoidB == 0.0);
// }

// double OpenCvSvm::classifyWithOptimalThreshold(const gsl::span<const float> sample) const
// {
//     return classificationProbability(sample) > m_optimalProbabilityThreshold ? 1.0 : 0.0;
// }

// void OpenCvSvm::setOptimalProbabilityThreshold(double optimalThreshold)
// {
//     m_optimalProbabilityThreshold = optimalThreshold;
// }

// bool OpenCvSvm::canClassifyWithOptimalThreshold() const
// {
//     return std::isnan(m_optimalProbabilityThreshold);
// }

// void OpenCvSvm::calculateSigmoidParametrs(const dataset::Dataset<std::vector<float>, float>& trainingSet)
// {
//     if (isTrained())
//     {
//         std::vector<float> rawValues;
//         rawValues.reserve(trainingSet.size());
//         for (auto& sample : trainingSet.getSamples())
//         {
//             constexpr auto rowsPerSample = 1;
//             const auto sampleMat = cv::Mat(rowsPerSample, static_cast<int>(sample.size()), CV_32FC1, sample.begin()._Ptr);
//             cv::Mat result;
//             m_svm->predict(sampleMat, result, cv::ml::StatModel::Flags::RAW_OUTPUT);

//             rawValues.emplace_back(result.at<float>(0, 0));
//         }

//         auto lables = trainingSet.getLabels();
//         std::vector<int> transformed;
//         std::transform(lables.begin(), lables.end(), std::back_inserter(transformed), [](auto f) { return static_cast<int>(std::round(f));});

//         const auto sigmoidParameters = libSvm::utils::sigmoidTrain(static_cast<unsigned int>(trainingSet.size()),
//                                                                    rawValues,
//                                                                    gsl::span<const int>(transformed));
//         m_sigmoidA = sigmoidParameters.m_A;
//         m_sigmoidB = sigmoidParameters.m_B;
//         return;
//     }
//     throw UntrainedSvmClassifierException();
// }
// }}// namespace phd::svm
