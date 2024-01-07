#pragma once

#include <vector>
#include <filesystem>
#include "libDataset/Dataset.h"
#include "Feature.h"

// namespace cv {
// 	class Mat;
// 	class TermCriteria;
// }


namespace phd { namespace svm {

enum class KernelTypes
{
    Custom = -1,
    Linear = 0,
    Poly = 1,
    Rbf = 2,
    Sigmoid = 3,
    Chi2 = 4,
    Inter = 5,
    Rbf_custom = 10,
    RBF_SUM,
	RBF_SUM_DIV2,
	RBF_DIV,
	RBF_MAX,
	RBF_MIN,
    RBF_SUM_2_KERNELS,
    RBF_LINEAR,
    RBF_LINEAR_SINGLE,
    RBF_LINEAR_MAX,
    RBF_LINEAR_MIN,
    RBF_LINEAR_SUM_2_KERNELS,
    RBF_POLY_GLOBAL
};

enum class SvmTypes
{
    Unknown,
    CSvc,
    NuSvc,
    OneClass,
    EpsSvr,
    NuSvr
};

class ISvm
{
public:
    virtual ~ISvm() = default;

    virtual std::shared_ptr<ISvm> clone() = 0;
	
    virtual double getC() const = 0;
    virtual double getGamma() const = 0;
    virtual std::vector<double> getGammas() const = 0;
    virtual double getCoef0() const = 0;
    virtual double getDegree() const = 0;
    virtual double getNu() const = 0;
    virtual double getP() const = 0;
    virtual double getT() const = 0;

    virtual void setC(double value) = 0;
    virtual void setCoef0(double value) = 0;
    virtual void setDegree(double value) = 0;
    virtual void setGamma(double value) = 0;
    virtual void setGammas(const std::vector<double>& value) = 0;
    virtual void setNu(double value) = 0;
    virtual void setP(double value) = 0;
    virtual void setT(double value) = 0;  //RBF_POLY_GLOBAL
    virtual void setOptimalProbabilityThreshold(double optimalThreshold) = 0;

    virtual KernelTypes getKernelType() const = 0;
    virtual void setKernel(KernelTypes kernelType) = 0;
    virtual SvmTypes getType() const = 0;
    virtual void setType(SvmTypes svmType) = 0;

    virtual void save(const std::filesystem::path& filepath) = 0;

    virtual bool isTrained() const = 0;
    virtual bool canGiveProbabilityOutput() const = 0;
    virtual bool canClassifyWithOptimalThreshold() const = 0;

    virtual uint32_t getNumberOfKernelParameters(KernelTypes kernelType) const = 0;
    virtual uint32_t getNumberOfSupportVectors() const = 0;
    virtual std::vector<std::vector<float>> getSupportVectors() const = 0;

    virtual float classify(const gsl::span<const float> sample) const = 0;
    virtual float classifyHyperplaneDistance(const gsl::span<const float> sample) const = 0;
    virtual double classificationProbability(const gsl::span<const float> sample) const = 0;
    virtual double classifyWithOptimalThreshold(const gsl::span<const float> sample) const = 0;


    virtual std::unordered_map<int, int> classifyGroups(const dataset::Dataset<std::vector<float>, float>& dataWithGroups) const = 0;
    virtual std::unordered_map<int, float> classifyGroupsRawScores(const dataset::Dataset<std::vector<float>, float>& dataWithGroups) const = 0;

    virtual float classifyWithCertainty(const gsl::span<const float> sample) const = 0;
    
    virtual void train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool probabilityNeeded = false) = 0;

    //virtual void setTerminationCriteria(const cv::TermCriteria& value) = 0;
    //virtual cv::TermCriteria getTerminationCriteria() const = 0;

	virtual void setFeatureSet(const std::vector<svmComponents::Feature>& features, int numberOfFeatures) = 0;
	virtual const std::vector<svmComponents::Feature>& getFeatureSet() = 0;
};
}} // namespace phd::svm

