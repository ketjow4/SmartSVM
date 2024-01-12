#include "libDataset/Dataset.h"
#include "SvmAucprcMetric.h"
#include "BaseSvmChromosome.h"
#include "SvmAccuracyMetric.h"

namespace svmComponents
{
struct AucprResults
{
	AucprResults(ConfusionMatrix matrix, double aucValue, double optimalThreshold)
		: m_matrix(std::move(matrix))
		, m_aucValue(aucValue)
		, m_optimalThreshold(optimalThreshold)
	{
	}

	const ConfusionMatrix m_matrix;
	const double m_aucValue;
	const double m_optimalThreshold;
};

Metric SvmAucprcMetric::calculateMetric(const BaseSvmChromosome& individual,
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

	std::vector<std::pair<double, int>> probabilites;
	probabilites.reserve(targets.size());

	for (auto i = 0u; i < targets.size(); i++)
	{
		probabilites.emplace_back(std::make_pair(svmModel->classifyHyperplaneDistance(samples[i]),
		                                         static_cast<int>(targets[i])));
	}
	AucprResults result = aucpr(probabilites, negativeCount, positiveCount);
	svmModel->setOptimalProbabilityThreshold(result.m_optimalThreshold); //TODO think of other way of setting this
	return Metric(result.m_aucValue, result.m_matrix);
}

double SvmAucprcMetric::trapezoidArea(double x1, double x2, double y1, double y2) const
{
	auto base = std::fabs(x1 - x2);
	auto height = (y1 + y2) / 2.0;
	return height * base;
}

AucprResults SvmAucprcMetric::aucpr(std::vector<std::pair<double, int>>& probabilityTargetPair, int negativeCount, int positiveCount) const
{
	std::sort(probabilityTargetPair.begin(), probabilityTargetPair.end(), [](const auto& a, const auto& b)
	{
		return a.first > b.first;
	});

	double auc = 0;
	double previousProbability = -1;
	double falsePositive = 0;
	double truePositive = 0;
	double falseNegative = 0;
	double trueNegative = 0;
	double precisionPreviousIteration = 0;
	double recallPreviousIteration = 0;

	auto maxAccuracyForThreshold = 0.0;
	auto threshold = 0.0;
	ConfusionMatrix matrixWithOptimalThreshold(0u, 0u, 0u, 0u);

	for (const auto& pair : probabilityTargetPair)
	{
		auto [probability, label] = pair;

		label == m_positiveClassValue ? truePositive++ : falsePositive++;
		trueNegative = (negativeCount - falsePositive);
		falseNegative = (positiveCount - truePositive);
		
		if (probability != previousProbability)
		{
			auto precision = truePositive / (truePositive + falsePositive);
			auto recall = truePositive / (truePositive + falseNegative);
			auc += trapezoidArea(recall, recallPreviousIteration, precision, precisionPreviousIteration);
			previousProbability = probability;
			precisionPreviousIteration = precision;
			recallPreviousIteration = recall;
		}


		const auto accuracyForThreshold = static_cast<double>(truePositive + (negativeCount - falsePositive)) / static_cast<double>(negativeCount +
			positiveCount);
		if (accuracyForThreshold > maxAccuracyForThreshold)
		{
			matrixWithOptimalThreshold = ConfusionMatrix(static_cast<uint32_t>(truePositive), static_cast<uint32_t>((negativeCount - falsePositive)),
				static_cast<uint32_t>(falsePositive), static_cast<uint32_t>((positiveCount - truePositive)));
			maxAccuracyForThreshold = accuracyForThreshold;
			threshold = probability;
		}
	}
	auto precision = truePositive / (truePositive + falsePositive);
	auto recall = truePositive / (truePositive + falseNegative);
	auc += trapezoidArea(recall, recallPreviousIteration, precision, precisionPreviousIteration);

	return AucprResults(matrixWithOptimalThreshold, auc, threshold);
}
} // namespace svmComponents
