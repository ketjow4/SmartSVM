// #pragma once

// #include <opencv2/ml.hpp>
// #include "libSvm/ISvm.h"
// #include "libSvm/SvmExceptions.h"

// namespace phd { namespace svm
// {
// class OpenCvSvm : public ISvm
// {
//     static constexpr double Nan = std::numeric_limits<double>::signaling_NaN();
// public:
//     ~OpenCvSvm() override = default;
//     OpenCvSvm();
//     explicit OpenCvSvm(const filesystem::Path& filepath);
//     explicit OpenCvSvm(cv::Ptr<cv::ml::SVM> svm, 
//                        double sigmoidA = 0.0,
//                        double sigmoidB = 0.0,
//                        double optimalProbabilityThreshold = Nan);


//     void setGammas(const std::vector<double>& /*value*/) override { throw UnsupportedKernelTypeException(KernelTypes::Rbf_custom); }
//     std::vector<double> getGammas() const override { throw UnsupportedKernelTypeException(KernelTypes::Rbf_custom); }
//     double getC() const override;
//     double getGamma() const override;
//     double getCoef0() const override;
//     double getDegree() const override;
//     double getNu() const override;
//     double getP() const override;
//     void setC(double value) override;
//     void setCoef0(double value) override;
//     void setDegree(double value) override;
//     void setGamma(double value) override;
//     void setNu(double value) override;
//     void setP(double value) override;
//     KernelTypes getKernelType() const override;
//     void setKernel(KernelTypes kernelType) override;
//     SvmTypes getType() const override;
//     void setType(SvmTypes svmType) override;
//     void save(const filesystem::Path& filepath) override;
//     uint32_t getNumberOfKernelParameters(KernelTypes kernelType) const override;
//     uint32_t getNumberOfSupportVectors() const override;
//     cv::Mat getSupportVectors() const override;

//     float classify(const gsl::span<const float> sample) const override;
//     void train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool probabilityNeeded = false) override;

//     double classificationProbability(const gsl::span<const float> sample) const override;
//     void setTerminationCriteria(const cv::TermCriteria& value) override;
//     cv::TermCriteria getTerminationCriteria() const override;
//     bool isTrained() const override;
//     bool canGiveProbabilityOutput() const override;

//     double classifyWithOptimalThreshold(const gsl::span<const float> sample) const override;
//     void setOptimalProbabilityThreshold(double optimalThreshold) override;
//     bool canClassifyWithOptimalThreshold() const override;

//     std::unordered_map<int, int> classifyGroups(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const
//     {
//         throw std::runtime_error("Classyfing groups not implemented in OpenCvSvm");
//     }

//     std::unordered_map<int, float> classifyGroupsRawScores(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const override
//     {
//         throw std::runtime_error("classifyGroupsRawScores not implemented in OpenCvSvm");
//     }

//     void train(cv::Ptr<cv::ml::TrainData> trainingSet) const;
//     void calculateSigmoidParametrs(const dataset::Dataset<std::vector<float>, float>& trainingSet);
//     cv::Ptr<cv::ml::TrainData> createTrainingData(const dataset::Dataset<std::vector<float>, float>& trainingSet);

// 	virtual void setFeatureSet(const std::vector<svmComponents::Feature>& features, int numberOfFeatures) override;
// 	virtual const std::vector<svmComponents::Feature>& getFeatureSet() override;

//     float classifyHyperplaneDistance(const gsl::span<const float> sample) const override;

//     float classifyWithCertainty(const gsl::span<const float> /*sample*/) const override
//     {
//         throw std::runtime_error("Not implemented EnsembleSVM classifyWithCertainty");
//     }

//     double getT() const override { throw std::runtime_error("Not implemented"); }
//     void setT(double /*value*/) override { throw std::runtime_error("Not implemented"); }

// private:
//     double sigmoidPredict(float decisionValue) const;

// 	void loadFeatureSet(std::string featuresDelimeted);

// public:
//     std::shared_ptr<ISvm> clone() override
//     {
//         throw std::runtime_error("Not implemented clone in OpenCvSvm");
//     }

    
// private:
//     cv::Ptr<cv::ml::SVM> m_svm;
//     double m_sigmoidA;
//     double m_sigmoidB;
//     double m_optimalProbabilityThreshold;
// 	std::vector<svmComponents::Feature> m_featureSet;
// 	int m_numberOfFeatures;

//     const int m_rowsPerSample = 1;
// };

// inline double OpenCvSvm::getC() const
// {
//     return m_svm->getC();
// }

// inline double OpenCvSvm::getGamma() const
// {
//     return m_svm->getGamma();
// }

// inline double OpenCvSvm::getCoef0() const
// {
//     return m_svm->getCoef0();
// }

// inline double OpenCvSvm::getDegree() const
// {
//     return m_svm->getDegree();
// }

// inline double OpenCvSvm::getNu() const
// {
//     return m_svm->getNu();
// }

// inline double OpenCvSvm::getP() const
// {
//     return m_svm->getP();
// }

// inline void OpenCvSvm::setC(double value)
// {
//     if (value > 0.0)
//     {
//         m_svm->setC(value);
//     }
//     else
//     {
//         throw ValueNotPositiveException("C");
//     }
// }

// inline void OpenCvSvm::setCoef0(double value)
// {
//     if (value >= 0.0)
//     {
//         m_svm->setCoef0(value);
//     }
//     else
//     {
//         throw ValueNotPositiveException("Coef0");
//     }
// }

// inline void OpenCvSvm::setDegree(double value)
// {
//     if (value > 0.0)
//     {
//         m_svm->setDegree(value);
//     }
//     else
//     {
//         throw ValueNotPositiveException("Degree");
//     }
// }

// inline void OpenCvSvm::setGamma(double value)
// {
//     if (value > 0.0)
//     {
//         m_svm->setGamma(value);
//     }
//     else
//     {
//         throw ValueNotPositiveException("Gamma");
//     }
// }

// inline void OpenCvSvm::setNu(double value)
// {
//     const auto minValue = 0.0;
//     const auto maxValue = 1.0;
//     if (value > minValue && value < maxValue)
//     {
//         m_svm->setNu(value);
//     }
//     else
//     {
//         throw ValueNotInRange("Nu", value, minValue, maxValue);
//     }
// }

// inline void OpenCvSvm::setP(double value)
// {
//     if (value > 0.0)
//     {
//         m_svm->setP(value);
//     }
//     else
//     {
//         throw ValueNotPositiveException("P");
//     }
// }
// }}// namespace phd::svm
