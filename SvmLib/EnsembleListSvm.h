#pragma once

//#include <opencv2/ml.hpp>

#include "ISvm.h"
#include "SvmExceptions.h"
#include "libSvmInternal.h"
#include "libSvmImplementation.h"
#include "ExtraTreeWrapper.h"

namespace phd { namespace svm
{
class ListNodeSvm
{
public:
	ListNodeSvm()
		: m_next(nullptr)
		, m_svm(nullptr)
	{
	}
	
	ListNodeSvm(std::shared_ptr<ListNodeSvm> next, std::shared_ptr<phd::svm::ISvm> svm)
		: m_next(next)
		, m_svm(svm)
	{
	}

	ListNodeSvm(std::shared_ptr<phd::svm::ISvm> svm)
		: m_next(nullptr)
		, m_svm(svm)
	{
	}

	std::shared_ptr<ListNodeSvm> m_next;
	std::shared_ptr<phd::svm::ISvm> m_svm;
};

class EnsembleListSvm : public phd::svm::ISvm
{
public:
	//std::shared_ptr<ExtraTreeWrapper> m_treeEndNode;
	
	EnsembleListSvm(std::shared_ptr<phd::svm::ListNodeSvm> list, int length);

	EnsembleListSvm(std::shared_ptr<phd::svm::ListNodeSvm> list, int length, bool newClassification);

	explicit EnsembleListSvm(const std::filesystem::path& filepath, bool newClassification = false);

	explicit EnsembleListSvm(const std::filesystem::path& filepath, bool newClassification, bool SESVM_AS_LAST);

	void setClassificationScheme(bool newClassification)
	{
		newClassificationScheme = newClassification;
	}

	double getC() const override
	{
		return 0;
	}

	double getGamma() const override
	{
		return 0;
	}

	double getCoef0() const override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	double getDegree() const override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	double getNu() const override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	double getP() const override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	double getT() const override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setC(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setCoef0(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setDegree(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setGamma(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setGammas(const std::vector<double>& /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setNu(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setP(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setT(double /*value*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setOptimalProbabilityThreshold(double /*optimalThreshold*/) override
	{
		; //do nothing for now
	}

	phd::svm::KernelTypes getKernelType() const override;

	void setKernel(phd::svm::KernelTypes /*kernelType*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	phd::svm::SvmTypes getType() const override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void setType(phd::svm::SvmTypes /*svmType*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	void save(const std::filesystem::path& /*filepath*/) override;

	bool isTrained() const override;

	bool canGiveProbabilityOutput() const override;

	bool canClassifyWithOptimalThreshold() const override;

	uint32_t getNumberOfKernelParameters(phd::svm::KernelTypes /*kernelType*/) const override;

	uint32_t getNumberOfSupportVectors() const override;

	std::vector<uint32_t> getNodesNumberOfSupportVectors() const;

	std::vector<std::vector<float>> getSupportVectorsOfLastNode();

	std::vector<std::vector<float>> getSupportVectors() const override;

	struct ResultAndThresholds
	{
		ResultAndThresholds(double resutls, double positiveThreshold, double negativeThreshold)
			: results(resutls)
			, positiveThreshold(positiveThreshold)
			, negativeThreshold(negativeThreshold)
		{
		}

		double results;
		double positiveThreshold;
		double negativeThreshold;
	};

	void unceratinClosestDistance(float& result, std::vector<ResultAndThresholds> history) const;
	void unceratinLargestDistance(float& result, std::vector<ResultAndThresholds> history) const;

	void unceratinVoting(float& result, std::vector<ResultAndThresholds> history) const;

	float classify(const gsl::span<const float> sample) const override;

	float classifyAll(const gsl::span<const float> sample) const;


	std::pair<float, int> LastNodeSchemeAndNode(const gsl::span<const float> sample) const;
	
	float DasvmScheme(const gsl::span<const float> sample) const;
	float LastNodeScheme(const gsl::span<const float> sample) const;
	std::pair<float, int> NewScheme(const gsl::span<const float> sample) const;
	std::pair<float, int> classifyWithNode(const gsl::span<const float> sample) const;

	//used for visualizations 
	float classifyWithCertainty(const gsl::span<const float> sample) const override;


	std::unordered_map<int, int> classifyGroups(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const
	{
		throw std::exception("Classyfing groups not implemented in EnsembleListSvm");
	}

	std::unordered_map<int, float> classifyGroupsRawScores(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const override
	{
		throw std::exception("classifyGroupsRawScores not implemented in EnsembleListSvm");
	}

	float classifyHyperplaneDistance(const gsl::span<const float> sample) const override;

	double classificationProbability(const gsl::span<const float> /*sample*/) const override
	{
		throw std::exception("Not implemented EnsembleListSvm classificationProbability");
	}

	double classifyWithOptimalThreshold(const gsl::span<const float> sample) const override;

	void train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool /*probabilityNeeded*/) override;

	// void setTerminationCriteria(const cv::TermCriteria& /*value*/) override
	// {
	// 	throw std::exception("Not implemented EnsembleListSvm");
	// }

	// cv::TermCriteria getTerminationCriteria() const override
	// {
	// 	throw std::exception("Not implemented EnsembleListSvm");
	// }

	void setFeatureSet(const std::vector<svmComponents::Feature>& /*features*/, int /*numberOfFeatures*/) override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	const std::vector<svmComponents::Feature>& getFeatureSet() override
	{
		throw std::exception("Not implemented EnsembleListSvm");
	}

	std::shared_ptr<ISvm> clone() override
	{
		return std::make_shared<EnsembleListSvm>(*this); //not real deep copy !!!
	}

	std::vector<double> getGammas() const override
	{
		throw std::exception("Not supported in EnsembleListSvm");
	}
	std::shared_ptr<phd::svm::ListNodeSvm> root;

	int list_length;
	int id;
	bool newClassificationScheme;
	std::vector<double> m_levelWeights;
};
}} // namespace phd::svm
