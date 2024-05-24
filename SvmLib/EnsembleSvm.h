#pragma once

//#include <opencv2/ml.hpp>
#include "ISvm.h"
#include "SvmExceptions.h"
#include "libSvmImplementation.h"

namespace phd { namespace svm
{
class EnsembleSvm : public ISvm
{
	static constexpr double Nan = std::numeric_limits<double>::signaling_NaN();
public:
	~EnsembleSvm() override;
	//EnsembleSvm(libSvmImplementation& linear,
	//libSvmImplementation& rbf);

	EnsembleSvm(std::vector<libSvmImplementation*> svms);

	double getC() const override;
	double getGamma() const override;
	std::vector<double> getGammas() const;
	double getCoef0() const override;
	double getDegree() const override;
	double getNu() const override;
	double getP() const override;
	void setC(double value) override;
	void setCoef0(double value) override;
	void setDegree(double value) override;
	void setGamma(double value) override;
	void setGammas(const std::vector<double>& value) override;
	void setNu(double value) override;
	void setP(double value) override;
	KernelTypes getKernelType() const override;
	void setKernel(KernelTypes kernelType) override;
	SvmTypes getType() const override;
	void setType(SvmTypes svmType) override;
	void save(const std::filesystem::path& filepath) override;
	uint32_t getNumberOfKernelParameters(KernelTypes kernelType) const override;
	uint32_t getNumberOfSupportVectors() const override;
	std::vector<std::vector<float>> getSupportVectors() const override;

	float classify(const gsl::span<const float> sample) const override;
	void train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool probabilityNeeded = false) override;

	double classificationProbability(const gsl::span<const float> sample) const override;
	//void setTerminationCriteria(const cv::TermCriteria& value) override;
	//cv::TermCriteria getTerminationCriteria() const override;
	bool isTrained() const override;
	bool canGiveProbabilityOutput() const override;

	double classifyWithOptimalThreshold(const gsl::span<const float> sample) const override;
	void setOptimalProbabilityThreshold(double optimalThreshold) override;
	bool canClassifyWithOptimalThreshold() const override;

	void setFeatureSet(const std::vector<svmComponents::Feature>& features, int numberOfFeatures) override;
	const std::vector<svmComponents::Feature>& getFeatureSet() override;

	float classifyHyperplaneDistance(const gsl::span<const float> sample) const override;

	double getT() const override { throw std::runtime_error("Not implemented"); }
	void setT(double /*value*/) override { throw std::runtime_error("Not implemented"); }

	std::unordered_map<int, int> classifyGroups(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const
	{
		throw std::runtime_error("Classyfing groups not implemented in EnsembleSvm");
	}

	std::unordered_map<int, float> classifyGroupsRawScores(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const override
	{
		throw std::runtime_error("classifyGroupsRawScores not implemented in EnsembleSvm");
	}


	bool m_optimalThresholdSet;
	std::vector<libSvmImplementation*> m_svms;

private:
	static std::vector<svm_node> convertSample(gsl::span<const float> sample)
	{
		std::vector<svm_node> node(sample.size() + 1);
		for (auto i = 0u; i < sample.size(); i++)
		{
			node[i].index = i;
			node[i].value = sample[i];
		}
		node[sample.size()].index = -1;
		node[sample.size()].value = 0;

		return node;
	}

	
	//void setupSupportVector(const dataset::Dataset<std::vector<float>, float>& trainingSet);

public:
	float classifyWithCertainty(const gsl::span<const float> /*sample*/) const override
	{
		throw std::runtime_error("Not implemented EnsembleSVM classifyWithCertainty");
	}

	std::shared_ptr<ISvm> clone() override
	{
		throw std::runtime_error("Not implemented clone EnsembleSvm");
	}
private:
	double m_optimalProbabilityThreshold;
	std::vector<svmComponents::Feature> m_featureSet;
	int m_numberOfFeatures;
};

inline EnsembleSvm::~EnsembleSvm()
{
}

inline EnsembleSvm::EnsembleSvm(std::vector<libSvmImplementation*> svms)
	: m_svms(svms)
{
	m_optimalProbabilityThreshold = 0;
}

inline std::vector<double> EnsembleSvm::getGammas() const
{
	throw std::runtime_error("Not implemented");
}

inline KernelTypes EnsembleSvm::getKernelType() const
{
	return KernelTypes::Custom;
}

inline void EnsembleSvm::setKernel(KernelTypes /*kernelType*/)
{
	throw std::runtime_error("Not implemented");
}

inline SvmTypes EnsembleSvm::getType() const
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setType(SvmTypes /*svmType*/)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::save(const std::filesystem::path& filepath)
{
	for (auto i = 0u; i < m_svms.size(); ++i)
	{
		m_svms[i]->save(filepath.string() + "__ensemble__" + std::to_string(i));
	}
}

inline uint32_t EnsembleSvm::getNumberOfKernelParameters(KernelTypes /*kernelType*/) const
{
	throw std::runtime_error("Not implemented");
}

inline uint32_t EnsembleSvm::getNumberOfSupportVectors() const
{
	auto sv_number = 0;
	for (auto i = 0u; i < m_svms.size(); ++i)
	{
		sv_number += m_svms[i]->getNumberOfSupportVectors();
	}
	return sv_number;
}

inline std::vector<std::vector<float>> EnsembleSvm::getSupportVectors() const
{
	auto a = m_svms[0]->getSupportVectors();
	for (auto i = 1u; i < m_svms.size(); ++i)
	{
		auto b = m_svms[i]->getSupportVectors();
		a.insert(a.end(), b.begin(),b.end());
	}
	
	return a;
}

inline float EnsembleSvm::classify(const gsl::span<const float> /*sample*/) const
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::train(const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/, bool /*probabilityNeeded*/)
{
	throw std::runtime_error("Not implemented");
}

inline double EnsembleSvm::classificationProbability(const gsl::span<const float> /*sample*/) const
{
	throw std::runtime_error("Not implemented");
}

inline bool EnsembleSvm::isTrained() const
{
	return true;
}

inline bool EnsembleSvm::canGiveProbabilityOutput() const
{
	return false;
}

inline double EnsembleSvm::classifyWithOptimalThreshold(const gsl::span<const float> sample) const
{
	return classifyHyperplaneDistance(sample) > m_optimalProbabilityThreshold ? 1.0 : 0.0;
}

inline void EnsembleSvm::setOptimalProbabilityThreshold(double optimalThreshold)
{
	m_optimalProbabilityThreshold = optimalThreshold;
}

inline bool EnsembleSvm::canClassifyWithOptimalThreshold() const
{
	return true;
}

inline void EnsembleSvm::setFeatureSet(const std::vector<svmComponents::Feature>& /*features*/, int /*numberOfFeatures*/)
{
	throw std::runtime_error("Not implemented");
}

inline const std::vector<svmComponents::Feature>& EnsembleSvm::getFeatureSet()
{
	throw std::runtime_error("Not implemented");
}

inline float EnsembleSvm::classifyHyperplaneDistance(const gsl::span<const float> sample) const
{
	float value = 0.0;
	for( auto& svm : m_svms)
	{
		value += svm->classifyHyperplaneDistance(sample);
	}
	return value;

	/*auto line = linear->classifyHyperplaneDistance(sample);
	auto rbf_answer = rbf->classifyHyperplaneDistance(sample);*/

	/*if (line > maxLine)
		maxLine = line;
	if (line < minLine)
		minLine = line;
	if (rbf_answer > maxLine)
		maxRBF = rbf_answer;
	if (rbf_answer < minRBF)
		minRBF = rbf_answer;*/

	//std::cout << "Line min/max: " << minLine << "/" << maxLine << "   RBF min/max: " << minRBF << "/" << maxRBF << "\n";
	/*std::cout << "RBF: " << rbf.classifyHyperplaneDistance(sample) << "       Linear: " << linear.classifyHyperplaneDistance(sample) << "\n";*/
	//if (std::abs(rbf.classifyHyperplaneDistance(sample)) > 0.5)
	/*{
		return linear->classifyHyperplaneDistance(sample) +  rbf->classifyHyperplaneDistance(sample);
	}*/
	//else
	//	return linear.classifyHyperplaneDistance(sample);
}

inline double EnsembleSvm::getC() const
{
	throw std::runtime_error("Not implemented");
}

inline double EnsembleSvm::getGamma() const
{
	throw std::runtime_error("Not implemented");
}

inline double EnsembleSvm::getCoef0() const
{
	throw std::runtime_error("Not implemented");
}

inline double EnsembleSvm::getDegree() const
{
	throw std::runtime_error("Not implemented");
}

inline double EnsembleSvm::getNu() const
{
	throw std::runtime_error("Not implemented");
}

inline double EnsembleSvm::getP() const
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setC(double)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setCoef0(double)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setDegree(double)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setGamma(double)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setGammas(const std::vector<double>&)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setNu(double)
{
	throw std::runtime_error("Not implemented");
}

inline void EnsembleSvm::setP(double)
{
	throw std::runtime_error("Not implemented");
}
}} // namespace phd::svm
