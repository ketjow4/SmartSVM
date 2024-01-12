

#include "libDataset/Dataset.h"
#include "SvmAucMetric.h"
#include "BaseSvmChromosome.h"
#include "SvmAccuracyMetric.h"
#include <fstream>
#include <set>

#include "libLogger/loguru.hpp"
#include "SvmLib/EnsembleListSvm.h"
#include "SvmLib/VotingEnsemble.h"

namespace svmComponents
{

struct AucResults
{
    AucResults(ConfusionMatrix matrix, double aucValue, double optimalThreshold)
        : m_matrix(std::move(matrix))
        , m_aucValue(aucValue)
        , m_optimalThreshold(optimalThreshold)
    {
    }

    const ConfusionMatrix m_matrix;
    const double m_aucValue;
    const double m_optimalThreshold;
};

Metric SvmAucMetric::calculateMetric(const BaseSvmChromosome& individual,
                                     const dataset::Dataset<std::vector<float>, float>& testSamples) const
{
    auto svmModel = individual.getClassifier();
    auto targets = testSamples.getLabels();
    auto samples = testSamples.getSamples();

    auto positiveCount = static_cast<unsigned int>(std::count_if(targets.begin(), targets.end(),
                                                                 [](const auto& target)
                                                                 {
                                                                     return target == m_positiveClassValue;
                                                                 }));
    auto negativeCount = static_cast<unsigned int>(samples.size() - positiveCount);



    if(testSamples.hasGroups())
    {
        LOG_F(WARNING, "Handling groups in AUC might be wrong due to no optimal threshold set");
        return handleGroups(individual, testSamples, false);
    }

    std::vector<std::pair<double, int>> probabilites;

    if (m_parallel)
    {
        probabilites.resize(targets.size());
#pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(targets.size()); i++)
        {
            auto temp_cl = svmModel->classifyHyperplaneDistance(samples[i]);
            probabilites[i] = std::make_pair(temp_cl, static_cast<int>(targets[i]));
        }
    }
    else
    {
        probabilites.reserve(targets.size());
        for (auto i = 0; i < targets.size(); i++)
        {
            probabilites.emplace_back(std::make_pair(svmModel->classifyHyperplaneDistance(samples[i]),
                static_cast<int>(targets[i])));
        }
    }
    LOG_F(WARNING, "AUC calculation does not set threshold, fix implementation");
    AucResults result = auc(probabilites, negativeCount, positiveCount);    
	//svmModel->setOptimalProbabilityThreshold(result.m_optimalThreshold);  //TODO think of other way of setting this
    return Metric(result.m_aucValue, ConfusionMatrix(individual, testSamples));
}

Metric SvmAucMetric::calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const
{
    auto svmModel = individual.getClassifier();
    auto targets = testSamples.getLabels();
    auto samples = testSamples.getSamples();

    auto positiveCount = static_cast<unsigned int>(std::count_if(targets.begin(), targets.end(),
        [](const auto& target)
        {
            return target == m_positiveClassValue;
        }));
    auto negativeCount = static_cast<unsigned int>(samples.size() - positiveCount);


    if (testSamples.hasGroups())
    {
    	return handleGroups(individual, testSamples, isTestSet);
    }

    std::vector<std::pair<double, int>> probabilites;

    if (m_parallel)
    {
	    probabilites.resize(targets.size());
	    auto iterationCount = static_cast<long long>(targets.size());

#pragma omp parallel for
	    for (long long i = 0; i < iterationCount; i++)
	    {
		    auto temp_cl = svmModel->classifyHyperplaneDistance(samples[i]);
		    probabilites[i] = std::make_pair(temp_cl, static_cast<int>(targets[i]));
	    }
    }
    else
    {
	    probabilites.reserve(targets.size());
	    for (auto i = 0; i < targets.size(); i++)
	    {
		    probabilites.emplace_back(std::make_pair(svmModel->classifyHyperplaneDistance(samples[i]),
		                                             static_cast<int>(targets[i])));
	    }
    }
	
    AucResults result = auc(probabilites, negativeCount, positiveCount);
	if(isTestSet == false)
    {
		svmModel->setOptimalProbabilityThreshold(result.m_optimalThreshold);  //TODO think of other way of setting this
        return Metric(result.m_aucValue, result.m_matrix);
    }
    //in here need to recalculate confusion matrix with proper threshold
	if(!svmModel->canClassifyWithOptimalThreshold() && (dynamic_cast<phd::svm::EnsembleListSvm*>(svmModel.get()) == nullptr || dynamic_cast<phd::svm::VotingEnsemble*>(svmModel.get()) == nullptr))
	{
        LOG_F(WARNING, "AUC calculationthreshold is not set. Fix implementation!");
	}
	if(testSamples.empty())
	{
        return Metric(1, ConfusionMatrix(0,0,0,0));
	}
	if(m_parallel)
    {
		return Metric(result.m_aucValue, ConfusionMatrix(individual, testSamples));
    }
    else
    {
        return Metric(result.m_aucValue, ConfusionMatrix(individual, testSamples));
    }
}

double SvmAucMetric::trapezoidArea(double x1, double x2, double y1, double y2) const
{
    auto base = std::fabs(x1 - x2);
    auto height = (y1 + y2) / 2.0;
    return height * base;
}

AucResults SvmAucMetric::auc(std::vector<std::pair<double, int>>& probabilityTargetPair, int negativeCount, int positiveCount) const
{
    std::sort(probabilityTargetPair.begin(), probabilityTargetPair.end(), [](const auto& a, const auto& b)
    {
        return a.first > b.first;
    });

    double auc = 0;
	double previousProbability = -1;
    int falsePositive = 0;
    int truePositive = 0;
    int falsePositivePreviousIteration = 0;
    int truePositivesPreviousIteration = 0;

    auto maxF1ForThreshold = 0.0;
    auto threshold = 0.0;
    ConfusionMatrix matrixWithOptimalThreshold(0u,0u,0u,0u);

    for (const auto& pair : probabilityTargetPair)
    {
        auto [probability,label] = pair;

		if (probability != previousProbability)
		{
			auc += trapezoidArea(falsePositive, falsePositivePreviousIteration, truePositive, truePositivesPreviousIteration);
			previousProbability = probability;
			falsePositivePreviousIteration = falsePositive;
			truePositivesPreviousIteration = truePositive;
		}

        label == m_positiveClassValue ? truePositive++ : falsePositive++;

        auto cm = ConfusionMatrix(truePositive, (negativeCount - falsePositive),falsePositive, (positiveCount - truePositive));


        double scoreForThreshold;
        if(m_metric == "ACC")
        {
            scoreForThreshold = cm.accuracy();
        }
        else if (m_metric == "MCC")
        {
            scoreForThreshold = cm.MCC();
        }
        else if (m_metric == "F1")
        {
            scoreForThreshold = cm.F1();
        }
        else
        {
            scoreForThreshold = cm.accuracy();
        }
       
        //const auto f1ForThreshold = cm.MCC();
        if(scoreForThreshold > maxF1ForThreshold)
        {
            matrixWithOptimalThreshold = cm;
            maxF1ForThreshold = scoreForThreshold;
            threshold = probability;
        }

    }
    //LOG_F(WARNING, "AUC threshold set to 0 (implemntation test)");
    auc += trapezoidArea(negativeCount, falsePositivePreviousIteration, positiveCount, truePositivesPreviousIteration);
    auc /= (static_cast<double>(positiveCount) * static_cast<double>(negativeCount));

    return AucResults(matrixWithOptimalThreshold, auc, threshold);
}

Metric SvmAucMetric::handleGroups(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const
{
    auto svmModel = individual.getClassifier();
    auto answers = svmModel->classifyGroupsRawScores(testSamples);

    auto labels = testSamples.getLabels();
    auto groups = testSamples.getGroups();


    std::map<int, std::unordered_set<int>> groups_labels;

    groups_labels[0] = std::unordered_set<int>();
    groups_labels[1] = std::unordered_set<int>();

    for(auto i = 0u; i < labels.size(); ++i)
    {
	    if(labels[i] == 0)
	    {
            groups_labels[0].emplace(static_cast<int>(groups[i]));
	    }
        else
        {
            groups_labels[1].emplace(static_cast<int>(groups[i]));
        }
    }

    auto negativeCount = static_cast<int>(groups_labels[0].size());
    auto positiveCount = static_cast<int>(groups_labels[1].size());

    std::vector<std::pair<double, int>> probabilites;

    for (auto [group, answer] : answers)
    {
        if (auto iter = std::find(groups.begin(), groups.end(), group); iter != groups.end())
        {
            auto index = std::distance(groups.begin(), iter);
            probabilites.emplace_back(answer, static_cast<int>(labels[index]));
        }
        else
        {
            LOG_F(ERROR, std::string("Group with ID: " + std::to_string(group) + " not found in dataset during AUC calculation").c_str());
            throw std::exception("Group not found in dataset during AUC calculation");
        }
    }

    AucResults result = auc(probabilites, negativeCount, positiveCount);

    if (isTestSet == false)
    {
        svmModel->setOptimalProbabilityThreshold(result.m_optimalThreshold);  //TODO think of other way of setting this
        return Metric(result.m_aucValue, result.m_matrix);
    }

    return Metric(result.m_aucValue, ConfusionMatrix(individual, testSamples));
}
} // namespace svmComponents
