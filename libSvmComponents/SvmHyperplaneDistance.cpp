
#include "BaseSvmChromosome.h"
#include "libSvmComponents/SvmHyperplaneDistance.h"

#include <fstream>


#include "libLogger/loguru.hpp"
#include "SvmLib/SvmLibImplementation.h"
#include "SvmAccuracyMetric.h"
#include "SvmAucMetric.h"
#include <iomanip>
#include "libTime/TimeUtils.h"

#pragma warning (disable: 4189)

namespace svmComponents
{

struct SvmAnswer
{
	SvmAnswer(double negativeClassAnswer,
	          double positiveClassAnswer,
	          double sumAnswer,
	          int target,
	          int i)
		: m_negativeClassAnswer(negativeClassAnswer)
		, m_positiveClassAnswer(positiveClassAnswer)
		, m_sumAnswer(sumAnswer)
		, m_target(target)
		, m_i(i)
	{
	}

	double m_negativeClassAnswer;
	double m_positiveClassAnswer;
	double m_sumAnswer;
	int m_target;
	int m_i;
};


Metric SvmHyperplaneDistance::calculateMetric(const BaseSvmChromosome& /*individual*/,
                                              const dataset::Dataset<std::vector<float>, float>& /*testSamples*/) const
{
	LOG_F(WARNING, "SvmHyperplaneDistance calculation use deprecated function");
	return Metric(0, 0, ConfusionMatrix(0, 0, 0, 0));
 //   auto svmModel = individual.getClassifier();
 //   auto targets = testSamples.getLabels();
 //   auto samples = testSamples.getSamples();

	//constexpr auto negativeClassValue = 0;
	//constexpr auto positiveClassValue = 1;

	////hyperplane distance, classify result, true value
 //   std::vector<std::tuple<double, int, int, int>> results;
	//classifySet(svmModel, targets, samples, results);

	//auto correctNumber = 0u;
	//double max_distance = -1000000.0;
	//double min_distance = 1000000.0;
	//double max_distance_raw = -1000000.0;
	//double min_distance_raw = 1000000.0;
	//for(auto i = 0u; i < results.size(); ++i)
	//{
	//	if (negativeClassValue != std::get<2>(results[i])) //if classify != target
	//	{
	//		if (min_distance == 1000000.0)
	//		{
	//			min_distance = -(std::get<0>(results[i]) / std::get<0>(results[0]));
	//			min_distance_raw = -1000000.0; //remove empty space with no vectors
	//		}
	//		break;
	//	}
	//	correctNumber++;
	//	
	//	if(std::get<0>(results[i]) < 0)
	//	{
	//		min_distance = -((std::get<0>(results[i]) / std::get<0>(results[0])) + (std::get<0>(results[i+1]) / std::get<0>(results[0]))) / 2;
	//	}
	//	else //if threshold is bigger than 0
	//	{
	//		min_distance = (std::get<0>(results[i]) / std::get<0>(results[results.size() - 1]) + (std::get<0>(results[i + 1]) / std::get<0>(results[results.size() - 1]))) / 2;
	//	}
	//	min_distance_raw = (std::get<0>(results[i]) + std::get<0>(results[i + 1])) / 2;
	//}

	//for (auto i = results.size() - 1; i >= 0; --i)
	//{
	//	if (positiveClassValue != std::get<2>(results[i])) //if classify != target
	//	{
	//		if (max_distance == -1000000)
	//		{
	//			max_distance = std::get<0>(results[i]) / std::get<0>(results[results.size() - 1]);
	//			max_distance_raw = 1000000;  //remove empty space with no vectors
	//		}
	//		break;
	//	}
	//	correctNumber++;

	//	if(std::get<0>(results[i]) > 0)
	//	{
	//		max_distance = (std::get<0>(results[i]) / std::get<0>(results[results.size() - 1]) + std::get<0>(results[i-1]) / std::get<0>(results[results.size() - 1])) / 2;
	//	}
	//	else
	//	{
	//		max_distance = -(std::get<0>(results[i]) / std::get<0>(results[0]) + std::get<0>(results[i - 1]) / std::get<0>(results[0])) / 2;
	//	}
	//	max_distance_raw = (std::get<0>(results[i]) + std::get<0>(results[i - 1])) / 2;

	//	if(max_distance_raw == min_distance_raw && correctNumber == samples.size())
	//	{
	//		max_distance = min_distance;
	//		//min_distance_raw = max_distance_raw = 0;
	//		break;
	//	}
	//}
	//
	//SvmAccuracyMetric acc;
	//auto result = acc.calculateMetric(individual, testSamples);

 //   const auto correctlyClassifiedCertainPercent = static_cast<float>(correctNumber) / static_cast<float>(samples.size());

	//auto epsilon = 0.000001;
	//if (std::fabs(max_distance_raw - min_distance_raw) < epsilon && correctlyClassifiedCertainPercent != 1.0)
	//{
	//	//fix error with float epsilon by moving slightly the boundaries otherwise depending on error stacking visualization and classification can be wrong
	//	max_distance_raw += epsilon;
	//	min_distance_raw -= epsilon;
	//	max_distance += epsilon;
	//	min_distance -= epsilon;
	//}

	//
	//auto distanceFitness = 1 - (max_distance - min_distance) / 2;

	////setThresholds(svmModel, max_distance_raw, min_distance_raw, epsilon);
	//
	//if (m_multicriteriaOptimization)
	//{
	//	auto multiMetric = (distanceFitness + correctlyClassifiedCertainPercent) / 2;
	//	return Metric(multiMetric, result.m_confusionMatrix.value());
	//}
	//if(m_useDistance)
	//{
	//	return Metric(distanceFitness, result.m_confusionMatrix.value());
	//}
	//else
	//{
	//	return Metric(correctlyClassifiedCertainPercent, result.m_confusionMatrix.value());
	//}
}

//#pragma optimize("", off)
void SvmHyperplaneDistance::classifySet(std::shared_ptr<phd::svm::ISvm> svmModel,
                                        gsl::span<const float> targets,
                                        gsl::span<const std::vector<float>> samples,
                                        std::vector<SvmAnswer>& results) const
{
	results.reserve(targets.size());
	auto svm = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
	auto bias = svm->m_model->rho[0]; //only for binary classification
	auto max = 0.0;
	
	for (auto i = 0u; i < targets.size(); i++)
	{
		auto [pos, neg, label] = svm->classifyPositiveNegative(samples[i]);
		auto sumAnswer = pos + neg - bias; //as in SvmLibInternal bias is always substructed from sum
		//auto sumAnswer2 = svm->classifyHyperplaneDistance(samples[i]);

		/*if ((label == 0 && sumAnswer > 0) || (label == 1 && sumAnswer < 0))
		{
			sumAnswer = -sumAnswer;
		}*/

		/*if (auto diff =  std::fabs( sumAnswer2 - sumAnswer); diff > max)
		{
			max = diff;
		}*/
		
		results.emplace_back(SvmAnswer(neg, pos, sumAnswer, static_cast<int>(targets[i]), i));
		//results.emplace_back(SvmAnswer(0, 0, sumAnswer2, static_cast<int>(targets[i]), i));
	}

	//std::cout << "Max diff: " << std::setprecision(8) << max << "\n";

	std::sort(results.begin(), results.end(), [&](const SvmAnswer& a, const SvmAnswer& b) { return a.m_sumAnswer < b.m_sumAnswer;  });
}

//#pragma optimize("", on)

void SvmHyperplaneDistance::setThresholds(std::shared_ptr<phd::svm::ISvm> svmModel, 
	double& max_distance_raw, 
	double& min_distance_raw,
	double& max_distance_normalized,
	double& min_distance_normalized) const
{
	auto res = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());

	auto epsilon = 0.000001;
	//if (std::fabs(max_distance_raw - -*res->m_model->rho) < epsilon)
	if (std::fabs(max_distance_raw - *res->m_model->rho) < epsilon || std::fabs(max_distance_raw - -*res->m_model->rho) < epsilon) 
		//nieintuicyjny troche ten warunek, dlaczego trzeba sprawdzac roznice modulow (Wart. bezwglednych) tych dwoch liczb zeby to dzialalo i nie bylo bug'a z przejsciem 
	{
		max_distance_raw += 100 * epsilon;
		//std::cout << "The problem here max_distance_raw!!!!\n";
	}
	//if (std::fabs(min_distance_raw - -*res->m_model->rho) < epsilon)
	if (std::fabs(min_distance_raw - *res->m_model->rho) < epsilon || std::fabs(min_distance_raw - -*res->m_model->rho) < epsilon)
	{
		min_distance_raw -= 100 * epsilon;
		//std::cout << "The problem here  min_distance_raw!!!!\n";
	}
		
	res->setCertaintyThreshold(min_distance_raw, max_distance_raw, min_distance_normalized, max_distance_normalized);
}

//#pragma optimize("", off)


//original version of metric calculation
//Metric SvmHyperplaneDistance::calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples,
//                                              bool /*isTestSet*/) const
//{
//	auto svmModel = individual.getClassifier();
//	auto targets = testSamples.getLabels();
//	auto samples = testSamples.getSamples();
//
//	constexpr auto negativeClassValue = 0;
//	constexpr auto positiveClassValue = 1;
//
//	//hyperplane distance, classify result, true value
//	//std::vector<std::tuple<double, int, int, int>> results;
//	//classifySet(svmModel, targets, samples, results);
//
//	auto correctNumber = 0u;
//
//	
//	
//	for(auto i = 0u; i < samples.size(); ++i)
//	{
//		auto answer = svmModel->classifyWithCertainty(samples[i]);
//
//		if( answer == targets[i] )
//		{
//			correctNumber++;
//		}
//	}
//
//
//	//SvmAccuracyMetric acc;
//	//auto result = acc.calculateMetric(individual, testSamples);   //TODO think if this is needed
//	//ConfusionMatrix none = result.m_confusionMatrix.value();
//	ConfusionMatrix none(0, 0, 0, 0);
//	
//	//SvmAucMetric auc;
//	//auto aucResult = auc.calculateMetric(individual, testSamples, isTestSet);
//	
//
//	auto correctlyClassifiedCertainPercent = 1.00 * (static_cast<float>(correctNumber) / static_cast<float>(samples.size())); //  +0.0 * result.m_fitness;
//
//	//auto distanceFitness = 1 - ((max_distance - min_distance) / 2);
//	auto SvmLibModel = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
//	auto distanceFitness = 1 - ((SvmLibModel->getPositiveNormalizedCertainty() - SvmLibModel->getNegativeNormalizedCertainty()) / 2);
//
//	if(m_distanceSet) //special treatment when K grows 
//	{
//		auto previousFitness = 1 - (m_DistanceBestPositive - m_DistanceBestNegative) / 2;
//		if(m_mode == zeroOut)
//		{
//			if(distanceFitness < previousFitness) //previous better
//			{
//				return Metric(0, distanceFitness, none);
//			}
//		}
//		else if (m_mode == nonlinearDecrease)
//		{
//			if (distanceFitness < previousFitness) //previous better
//			{
//				auto difference = previousFitness - distanceFitness;
//				auto decrese = 1 / std::pow(1 + difference, 4);
//				
//				return Metric(correctlyClassifiedCertainPercent * decrese, distanceFitness, none);
//			}
//		}
//		else if (m_mode == boundryCheck)
//		{
//			if (std::fabs(previousFitness - distanceFitness) > 0.001) 
//			{
//				return Metric(0, distanceFitness, none);
//			}
//		}
//	}
//
//	//if(m_multicriteriaOptimization)
//	//{
//	//	auto multiMetric = (distanceFitness + correctlyClassifiedCertainPercent) / 2;
//	//	//std::cout << multiMetric << "\n";
//	//	return Metric(multiMetric, result.m_confusionMatrix.value());
//	//}
//	//if (m_useDistance)
//	//{
//	//	return Metric(distanceFitness, result.m_confusionMatrix.value());
//	//}
//	//else
//	{
//		return Metric(correctlyClassifiedCertainPercent, distanceFitness, none);
//	}
//}
//

//#pragma optimize("", off)

Metric SvmHyperplaneDistance::calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples,
	bool /*isTestSet*/) const
{
	auto svmModel = individual.getClassifier();
	auto targets = testSamples.getLabels();
	auto samples = testSamples.getSamples();

	constexpr auto negativeClassValue = 0;
	constexpr auto positiveClassValue = 1;

	auto positiveCount = static_cast<unsigned int>(std::count_if(targets.begin(), targets.end(),
		[&](const auto& target)
		{
			return target == positiveClassValue;
		}));
	auto negativeCount = static_cast<unsigned int>(samples.size() - positiveCount);

	//hyperplane distance, classify result, true value
	std::vector<SvmAnswer> results;
	classifySet(svmModel, targets, samples, results);

	auto svm = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
	auto bias = svm->m_model->rho[0]; //only for binary classification


	auto correctNumber = 0u;
	double max_distance = -1000000.0;
	double min_distance = 1000000.0;
	double max_distance_raw = -1000000.0;
	double min_distance_raw = 1000000.0;
	auto epsilon = 0.000001;

	double negativeClassDistance = 1000000.0;
	double positiveClassDistance = -1000000.0;

	//vector that contains pair of threshold and number of correct samples at that threshold;
	std::vector<std::pair<double, int>> negativeThrSamples;


	auto hard_threshold_percent = 0.2;
	auto hard_coded_threshold_neg = results[0].m_sumAnswer * hard_threshold_percent;
	auto hard_coded_threshold_pos = results[results.size() - 1].m_sumAnswer * hard_threshold_percent;

	int negativeCorrectNumber = 0;
	auto negativeIndex = 0u;
	for (auto i = 0u; i < results.size(); ++i)
	{
		if (results[i].m_sumAnswer > hard_coded_threshold_neg)
		{
			if (min_distance == 1000000.0)
			{
				min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				min_distance_raw = -1000000.0; //remove empty space with no vectors
				negativeClassDistance = -1000000.0;
			}
			break;
		}

		if (m_useBias && results[i].m_sumAnswer > -bias)
		{
			if (min_distance == 1000000.0)
			{
				min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				min_distance_raw = -1000000.0; //remove empty space with no vectors
				negativeClassDistance = -1000000.0;
			}
			else if (i > 1) //to calculate threshold
			{
				min_distance_raw = (results[i - 1].m_sumAnswer + results[i - 2].m_sumAnswer) / 2;
				negativeClassDistance = results[i - 1].m_negativeClassAnswer;
			}
			else if (i == 1)
			{
				min_distance_raw = results[0].m_sumAnswer;
				negativeClassDistance = results[0].m_negativeClassAnswer;
			}
			break;
		}


		//2 following vectors need to have different decision values in order to differentiate them
		if (i != 0 && results[i].m_sumAnswer == results[i - 1].m_sumAnswer)
		{
			if (negativeClassValue != results[i].m_target)
			{
				if (results[i].m_sumAnswer < 0)
				{
					min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				}
				else //if threshold is bigger than 0
				{
					min_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				}
				min_distance_raw = results[i].m_sumAnswer - epsilon;
				negativeClassDistance = results[i].m_negativeClassAnswer;
				break;
			}
			continue;
		}

		/*if (std::get<0>(results[i]) == 0)
			break;*/

		if (negativeClassValue != results[i].m_target) //if classify != target
		{
			if (min_distance == 1000000.0)
			{
				min_distance = -(results[i].m_sumAnswer / results[0].m_sumAnswer);
				min_distance_raw = -1000000.0; //remove empty space with no vectors
				negativeClassDistance = -1000000.0;
			}
			break;
		}
		if (results[i].m_sumAnswer != 0) //do not accept zero as valid answer because that means that example is so far away that we know nothing
		{
			correctNumber++;
			negativeCorrectNumber++;
			negativeIndex = i;
		}

		if (results[i].m_sumAnswer < 0)
		{
			min_distance = -((results[i].m_sumAnswer / results[0].m_sumAnswer) + (results[i + 1].m_sumAnswer / results[0].m_sumAnswer)) / 2;
		}
		else //if threshold is bigger than 0
		{
			min_distance = ((results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer) + (results[i + 1].m_sumAnswer / results[results.size() - 1].m_sumAnswer)) / 2;
		}
		min_distance_raw = (results[i].m_sumAnswer + results[i + 1].m_sumAnswer) / 2;
		negativeClassDistance = results[i].m_negativeClassAnswer;


		//negativeThrSamples.emplace_back(results[i].m_sumAnswer, correctNumber);

		if (m_distanceSet && m_mode == boundryCheck && min_distance >= m_DistanceBestNegative)
		{
			break;
		}
	}

	std::vector<std::pair<double, int>> positiveThrSamples;
	int positiveCorrectNumber = 0;
	auto positiveIndex = results.size() - 1;
	for (auto i = results.size() - 1; i >= 0; --i)
	{
		if (results[i].m_sumAnswer < hard_coded_threshold_pos)
		{
			if (max_distance == -1000000)
			{
				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				max_distance_raw = 1000000;  //remove empty space with no vectors
				positiveClassDistance = 1000000;
			}
			break;
		}


		if (m_useBias && results[i].m_sumAnswer < -bias)
		{
			if (max_distance == -1000000)
			{
				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				max_distance_raw = 1000000;  //remove empty space with no vectors
				positiveClassDistance = 1000000;
			}
			else if (i < results.size() - 2)
			{
				max_distance_raw = (results[i + 1].m_sumAnswer + results[i + 2].m_sumAnswer) / 2;
				positiveClassDistance = results[i + 1].m_positiveClassAnswer;
			}
			else if (i == results.size() - 1)
			{
				max_distance_raw = results[i].m_sumAnswer;
				positiveClassDistance = results[i].m_positiveClassAnswer;
			}
			break;
		}


		//2 following vectors need to have different decision values in order to differentiate them
		if (i != results.size() - 1 && results[i].m_sumAnswer == results[i + 1].m_sumAnswer)
		{
			if (positiveClassValue != results[i].m_target)
			{
				if (results[i].m_sumAnswer > 0)
				{
					max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				}
				else
				{
					max_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				}
				max_distance_raw = results[i].m_sumAnswer + epsilon;
				positiveClassDistance = results[i].m_positiveClassAnswer;
				break;
			}
			continue;
		}


		if (positiveClassValue != results[i].m_target) //if classify != target
		{
			if (max_distance == -1000000)
			{
				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				max_distance_raw = 1000000;  //remove empty space with no vectors
				positiveClassDistance = 1000000;
			}
			break;
		}

		if (results[i].m_sumAnswer != min_distance_raw && results[i].m_sumAnswer != 0)
		{
			correctNumber++;
			positiveCorrectNumber++;
			positiveIndex = i;
		}

		if (results[i].m_sumAnswer > 0)
		{
			max_distance = ((results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer) + results[i - 1].m_sumAnswer / results[results.size() - 1].m_sumAnswer) / 2;
		}
		else
		{
			max_distance = -(results[i].m_sumAnswer / results[0].m_sumAnswer + results[i - 1].m_sumAnswer / results[0].m_sumAnswer) / 2;
		}
		max_distance_raw = (results[i].m_sumAnswer + results[i - 1].m_sumAnswer) / 2;
		positiveClassDistance = results[i].m_positiveClassAnswer;

		//positiveThrSamples.emplace_back(results[i].m_sumAnswer, positiveCorrectNumber);

		if (max_distance_raw == min_distance_raw && correctNumber == samples.size())
		{
			max_distance = min_distance;
			//min_distance_raw = max_distance_raw = 0;
			break;
		}

		if (m_distanceSet && m_mode == boundryCheck && max_distance <= m_DistanceBestPositive)
		{
			break;
		}
	}

	//std::ofstream debugThrehsolds("D:\\ENSEMBLE_721_BIG_SETSdebug\\" + timeUtils::getTimestamp() + "_thresholds.txt");
	//debugThrehsolds << "Current neg, current pos\n";
	//debugThrehsolds << min_distance_raw << ", " << max_distance_raw << "\n";
	//debugThrehsolds << "Min and max\n";
	//debugThrehsolds << results[0].m_sumAnswer << ", "  << results[results.size() - 1].m_sumAnswer << "\n";
	//debugThrehsolds << "#neg ans, pos ans,  sum, target\n";
	//
	//std::vector<std::pair<double, double>> distanceAndSamples;
	//for(auto& sample : results)
	//{
	//	debugThrehsolds << sample.m_negativeClassAnswer << ", " << sample.m_positiveClassAnswer << ", " << sample.m_sumAnswer << ", " << sample.m_target << "\n";
	//}

	//for(auto negativeThr : negativeThrSamples)
	//{
	//	for(auto positiveThr : positiveThrSamples)
	//	{
	//		auto distance = positiveThr.first - negativeThr.first;
	//		auto samplesCovered = static_cast<double>(negativeThr.second + positiveThr.second) / static_cast<double>(samples.size());
	//		//distanceAndSamples.e
	//		debugThrehsolds << negativeThr.first << ", " << positiveThr.first << ", " << distance << ", " << samplesCovered << "\n";
	//	}
	//}


	auto correctlyClassifiedCertainPercent = 10.00 * (static_cast<float>(correctNumber) / static_cast<float>(samples.size()));

	////TODO fix by config
	//auto thresholdNumberOfVectors = std::min(100u, static_cast<unsigned>(testSamples.size()));

	//std::vector<double> mcc_scores;
	//if (correctNumber < thresholdNumberOfVectors)
	//{
	//	//set thresholds for max MCC that takes 100 vectors
	//	auto missingCount = thresholdNumberOfVectors - correctNumber;

	//	auto tp = static_cast<unsigned int>(results.size() - positiveIndex);
	//	auto tn = negativeIndex;
	//	auto fp = 0;
	//	auto fn = 0;

	//	for (auto j = positiveIndex - 1; j > positiveIndex - missingCount; j--)
	//	{
	//		if (results[j].m_target == 1)
	//		{
	//			tp++;
	//		}
	//		else
	//		{
	//			fn++;
	//		}
	//	}

	//	for (auto i = negativeIndex; i < negativeIndex + missingCount; i++)
	//	{
	//		if (results[i].m_target == 0)
	//		{
	//			tn++;
	//		}
	//		else
	//		{
	//			fp++;
	//		}

	//		ConfusionMatrix matrix(tp, tn, fp, fn);
	//		auto mcc = matrix.MCC();
	//		mcc_scores.emplace_back(mcc);

	//		auto posIndexCorresponding = positiveIndex - missingCount + 1 + (i - negativeIndex);

	//		if(posIndexCorresponding > results.size())
	//		{
	//			LOG_F(WARNING, "Optimization of MCC might have a bug, double check it in future");
	//			break;
	//		}

 //			if (results[posIndexCorresponding].m_target == 1)
	//		{
	//			tp--;
	//		}
	//		else
	//		{
	//			fn--;
	//		}
	//	}

	//	auto maxIndex = std::distance(mcc_scores.begin(), std::max_element(mcc_scores.begin(), mcc_scores.end()));


	//	//Setting up new thresholds and fitness value
	//	correctlyClassifiedCertainPercent = mcc_scores[maxIndex];
	//	
	//	min_distance_raw = results[negativeIndex + maxIndex].m_sumAnswer;
	//	min_distance = results[negativeIndex + maxIndex].m_sumAnswer / results[0].m_sumAnswer;
	//	negativeClassDistance = results[negativeIndex + maxIndex].m_negativeClassAnswer;

	//	max_distance_raw = results[positiveIndex - maxIndex].m_sumAnswer;
	//	max_distance = results[positiveIndex - maxIndex].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
	//	positiveClassDistance = results[positiveIndex - maxIndex].m_positiveClassAnswer;

	//	//LOG_F(INFO, "Mcc score 100 vectors: %f", correctlyClassifiedCertainPercent);
	//}

	


	/*auto correctlyClassifiedCertainPercent = static_cast<float>(positiveCorrectNumber) / static_cast<float>(positiveCount)
											+ static_cast<float>(negativeCorrectNumber) / static_cast<float>(negativeCount);*/

											//LOG_F(INFO, "Correctly classified percent %f", correctlyClassifiedCertainPercent);
											//LOG_F(INFO, "correctlyClassifiedCertainPercent %f", correctlyClassifiedCertainPercent);

	epsilon = 0.000001;
	if (std::fabs(max_distance_raw - min_distance_raw) < epsilon && correctlyClassifiedCertainPercent != 1.0)
	{
		//correctlyClassifiedCertainPercent = 0.0;
		//fix error with float epsilon by moving slightly the boundaries otherwise depending on error stacking visualization and classification can be wrong
		max_distance_raw += epsilon;
		min_distance_raw -= epsilon;
		max_distance += epsilon;
		min_distance -= epsilon;
	}

	if (std::isnan(max_distance))
	{
		LOG_F(WARNING, "Max distance was nan, fixing it to 0");
		max_distance = 0;
	}
	if (std::isnan(min_distance))
	{
		LOG_F(WARNING, "Min distance was nan, fixing it to 0");
		min_distance = 0;
	}

	//auto distanceFitness = 1 - ((max_distance - min_distance) / 2);

	setThresholds(svmModel, max_distance_raw, min_distance_raw, max_distance, min_distance);


	//if these threshold are not set they are not used and default value of -1111 is set for them
	if (m_useSingleClassPrediction)
	{
		auto res = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
		res->setClassCertaintyThreshold(negativeClassDistance, positiveClassDistance, negativeClassDistance, positiveClassDistance);
	}

	
	ConfusionMatrix none(0, 0, 0, 0);

	//auto distanceFitness = 1 - ((max_distance - min_distance) / 2);
	auto SvmLibModel = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
	auto distanceFitness = 1 - ((SvmLibModel->getPositiveNormalizedCertainty() - SvmLibModel->getNegativeNormalizedCertainty()) / 2);

	if (m_distanceSet) //special treatment when K grows 
	{
		auto previousFitness = 1 - (m_DistanceBestPositive - m_DistanceBestNegative) / 2;
		if (m_mode == zeroOut)
		{
			if (distanceFitness < previousFitness) //previous better
			{
				return Metric(0, distanceFitness, none);
			}
		}
		else if (m_mode == nonlinearDecrease)
		{
			if (distanceFitness < previousFitness) //previous better
			{
				auto difference = previousFitness - distanceFitness;
				auto decrese = 1 / std::pow(1 + difference, 4);

				return Metric(correctlyClassifiedCertainPercent * decrese, distanceFitness, none);
			}
		}
		else if (m_mode == boundryCheck)
		{
			if (std::fabs(previousFitness - distanceFitness) > 0.001)
			{
				return Metric(0, distanceFitness, none);
			}
		}
	}
	{
		return Metric(correctlyClassifiedCertainPercent, distanceFitness, none);
	}
}




//#pragma optimize("", off)

//void SvmHyperplaneDistance::calculateThresholds(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples)
//{
//	auto svmModel = individual.getClassifier();
//	auto targets = testSamples.getLabels();
//	auto samples = testSamples.getSamples();
//
//	constexpr auto negativeClassValue = 0;
//	constexpr auto positiveClassValue = 1;
//
//	//hyperplane distance, classify result, true value
//	std::vector<SvmAnswer> results;
//	classifySet(svmModel, targets, samples, results);
//
//	auto svm = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
//	auto bias = svm->m_model->rho[0]; //only for binary classification
//
//	
//	auto correctNumber = 0u;
//	double max_distance = -1000000.0;
//	double min_distance = 1000000.0;
//	double max_distance_raw = -1000000.0;
//	double min_distance_raw = 1000000.0;
//	auto epsilon = 0.000001;
//
//	//vector that contains pair of threshold and number of correct samples at that threshold;
//	std::vector<std::pair<double, int>> negativeThrSamples;
//	
//	for (auto i = 0u; i < results.size(); ++i)
//	{
//		//if (std::get<0>(results[i]) > -bias || std::get<0>(results[i]) > 0)
//		//{
//		//	if (min_distance == 1000000.0)
//		//	{
//		//		min_distance = -(std::get<0>(results[i]) / std::get<0>(results[0]));
//		//		min_distance_raw = -1000000.0; //remove empty space with no vectors
//		//	}
//		//	break;
//		//}
//
//		
//		//2 following vectors need to have different decision values in order to differentiate them
//		if (i != 0 && results[i].m_sumAnswer == results[i-1].m_sumAnswer)
//		{
//			if (negativeClassValue != results[i].m_target)
//			{
//				if (results[i].m_sumAnswer < 0)
//				{
//					min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
//				}
//				else //if threshold is bigger than 0
//				{
//					min_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
//				}
//				min_distance_raw = results[i].m_sumAnswer - epsilon;
//				break;
//			}
//			continue;
//		}
//		
//		/*if (std::get<0>(results[i]) == 0)
//			break;*/
//
//		if (negativeClassValue != results[i].m_target) //if classify != target
//		{
//			if (min_distance == 1000000.0)
//			{
//				min_distance = -(results[i].m_sumAnswer / results[0].m_sumAnswer);
//				min_distance_raw = -1000000.0; //remove empty space with no vectors
//			}
//			break;
//		}
//		if (results[i].m_sumAnswer != 0) //do not accept zero as valid answer because that means that example is so far away that we know nothing
//		{
//			correctNumber++;
//		}
//
//		if (results[i].m_sumAnswer < 0)
//		{
//			min_distance = -((results[i].m_sumAnswer / results[0].m_sumAnswer) + (results[i + 1].m_sumAnswer / results[0].m_sumAnswer)) / 2;
//		}
//		else //if threshold is bigger than 0
//		{
//			min_distance = ((results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer) + (results[i + 1].m_sumAnswer / results[results.size() - 1].m_sumAnswer)) / 2;
//		}
//		min_distance_raw = (results[i].m_sumAnswer + results[i + 1].m_sumAnswer) / 2;
//
//
//		//negativeThrSamples.emplace_back(results[i].m_sumAnswer, correctNumber);
//
//		if (m_distanceSet && m_mode == boundryCheck && min_distance >= m_DistanceBestNegative)
//		{
//			break;
//		}
//	}
//
//	std::vector<std::pair<double, int>> positiveThrSamples;
//	int positiveCorrectNumber = 0;
//	
//	for (auto i = results.size() - 1; i >= 0; --i)
//	{
//		//if (std::get<0>(results[i]) < -bias || std::get<0>(results[i]) < 0)
//		//{
//		//	if (max_distance == -1000000)
//		//	{
//		//		max_distance = std::get<0>(results[i]) / std::get<0>(results[results.size() - 1]);
//		//		max_distance_raw = 1000000;  //remove empty space with no vectors
//		//	}
//		//	break;
//		//}
//
//		
//		//2 following vectors need to have different decision values in order to differentiate them
//		if (i != results.size() - 1 && results[i].m_sumAnswer == results[i + 1].m_sumAnswer)
//		{
//			if (positiveClassValue != results[i].m_target)
//			{
//				if (results[i].m_sumAnswer > 0)
//				{
//					max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
//				}
//				else
//				{
//					max_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
//				}
//				max_distance_raw = results[i].m_sumAnswer + epsilon;
//				break;
//			}
//			continue;
//		}
//
//		
//		if (positiveClassValue != results[i].m_target) //if classify != target
//		{
//			if (max_distance == -1000000)
//			{
//				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
//				max_distance_raw = 1000000;  //remove empty space with no vectors
//			}
//			break;
//		}
//
//		if (results[i].m_sumAnswer != min_distance_raw && results[i].m_sumAnswer != 0)
//		{
//			correctNumber++;
//			positiveCorrectNumber++;
//		}
//
//		if (results[i].m_sumAnswer > 0)
//		{
//			max_distance = ((results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer) + results[i - 1].m_sumAnswer / results[results.size() - 1].m_sumAnswer) / 2;
//		}
//		else
//		{
//			max_distance = -(results[i].m_sumAnswer / results[0].m_sumAnswer + results[i - 1].m_sumAnswer / results[0].m_sumAnswer) / 2;
//		}
//		max_distance_raw = (results[i].m_sumAnswer + results[i-1].m_sumAnswer) / 2;
//
//
//		positiveThrSamples.emplace_back(results[i].m_sumAnswer, positiveCorrectNumber);
//		
//		if (max_distance_raw == min_distance_raw && correctNumber == samples.size())
//		{
//			max_distance = min_distance;
//			//min_distance_raw = max_distance_raw = 0;
//			break;
//		}
//
//		if (m_distanceSet && m_mode == boundryCheck && max_distance <= m_DistanceBestPositive)
//		{
//			break;
//		}
//	}
//
//	//std::ofstream debugThrehsolds("D:\\ENSEMBLE_566_2D_debug_threshold\\A1_501\\1\\" + timeUtils::getTimestamp() + "_thresholds.txt");
//	//debugThrehsolds << "Current neg, current pos\n";
//	//debugThrehsolds << min_distance_raw << ", " << max_distance_raw << "\n";
//	//debugThrehsolds << "Min and max\n";
//	//debugThrehsolds << std::get<0>(results[0]) << ", "  << std::get<0>(results[results.size() - 1]) << "\n";
//	//debugThrehsolds << "#neg thr, pos thr,  distance, samples covered percent\n";
//	//
//	//std::vector<std::pair<double, double>> distanceAndSamples;
//	//for(auto negativeThr : negativeThrSamples)
//	//{
//	//	for(auto positiveThr : positiveThrSamples)
//	//	{
//	//		auto distance = positiveThr.first - negativeThr.first;
//	//		auto samplesCovered = static_cast<double>(negativeThr.second + positiveThr.second) / static_cast<double>(samples.size());
//	//		//distanceAndSamples.e
//	//		debugThrehsolds << negativeThr.first << ", " << positiveThr.first << ", " << distance << ", " << samplesCovered << "\n";
//	//	}
//	//}
//
//	
//	auto correctlyClassifiedCertainPercent = 1.00 * (static_cast<float>(correctNumber) / static_cast<float>(samples.size()));
//
//	//LOG_F(INFO, "Correctly classified percent %f", correctlyClassifiedCertainPercent);
//	//LOG_F(INFO, "correctlyClassifiedCertainPercent %f", correctlyClassifiedCertainPercent);
//
//	epsilon = 0.000001;
//	if (std::fabs(max_distance_raw - min_distance_raw) < epsilon && correctlyClassifiedCertainPercent != 1.0)
//	{
//		//correctlyClassifiedCertainPercent = 0.0;
//		//fix error with float epsilon by moving slightly the boundaries otherwise depending on error stacking visualization and classification can be wrong
//		max_distance_raw += epsilon;
//		min_distance_raw -= epsilon;
//		max_distance += epsilon;
//		min_distance -= epsilon;
//	}
//
//	if (std::isnan(max_distance))
//	{
//		LOG_F(WARNING, "Max distance was nan, fixing it to 0");
//		max_distance = 0;
//	}
//	if (std::isnan(min_distance))
//	{
//		LOG_F(WARNING, "Min distance was nan, fixing it to 0");
//		min_distance = 0;
//	}
//
//	//auto distanceFitness = 1 - ((max_distance - min_distance) / 2);
//
//	setThresholds(svmModel, max_distance_raw, min_distance_raw, max_distance, min_distance);
//
//	/*auto result = calculateMetric(individual, testSamples, false);
//	if (result.m_fitness > 0.2)
//	{
//		std::cout << "Check what is happening\n";
//	}*/
//}
//



void SvmHyperplaneDistance::calculateThresholds(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples)
{
	auto svmModel = individual.getClassifier();
	auto targets = testSamples.getLabels();
	auto samples = testSamples.getSamples();

	constexpr auto negativeClassValue = 0;
	constexpr auto positiveClassValue = 1;

	auto positiveCount = static_cast<unsigned int>(std::count_if(targets.begin(), targets.end(),
		[&](const auto& target)
		{
			return target == positiveClassValue;
		}));
	auto negativeCount = static_cast<unsigned int>(samples.size() - positiveCount);

	//hyperplane distance, classify result, true value
	std::vector<SvmAnswer> results;
	classifySet(svmModel, targets, samples, results);

	auto svm = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
	auto bias = svm->m_model->rho[0]; //only for binary classification


	auto correctNumber = 0u;
	double max_distance = -1000000.0;
	double min_distance = 1000000.0;
	double max_distance_raw = -1000000.0;
	double min_distance_raw = 1000000.0;
	auto epsilon = 0.000001;

	double negativeClassDistance = 1000000.0;
	double positiveClassDistance = -1000000.0;

	//vector that contains pair of threshold and number of correct samples at that threshold;
	std::vector<std::pair<double, int>> negativeThrSamples;


	auto hard_threshold_percent = 0.2;
	auto hard_coded_threshold_neg = results[0].m_sumAnswer * hard_threshold_percent;
	auto hard_coded_threshold_pos = results[results.size()-1].m_sumAnswer * hard_threshold_percent;
	
	int negativeCorrectNumber = 0;
	auto negativeIndex = 0u;
	for (auto i = 0u; i < results.size(); ++i)
	{
		if(results[i].m_sumAnswer > hard_coded_threshold_neg)
		{
			if (min_distance == 1000000.0)
			{
				min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				min_distance_raw = -1000000.0; //remove empty space with no vectors
				negativeClassDistance = -1000000.0;
			}
			break;
		}
		
		if (m_useBias && results[i].m_sumAnswer > -bias )
		{
			if (min_distance == 1000000.0)
			{
				min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				min_distance_raw = -1000000.0; //remove empty space with no vectors
				negativeClassDistance = -1000000.0;
			}
			else if (i > 1) //to calculate threshold
			{
				min_distance_raw = (results[i - 1].m_sumAnswer + results[i - 2].m_sumAnswer) / 2;
				negativeClassDistance = results[i - 1].m_negativeClassAnswer;
			}
			else if (i == 1)
			{
				min_distance_raw = results[0].m_sumAnswer;
				negativeClassDistance = results[0].m_negativeClassAnswer;
			}
			break;
		}


		//2 following vectors need to have different decision values in order to differentiate them
		if (i != 0 && results[i].m_sumAnswer == results[i - 1].m_sumAnswer)
		{
			if (negativeClassValue != results[i].m_target)
			{
				if (results[i].m_sumAnswer < 0)
				{
					min_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				}
				else //if threshold is bigger than 0
				{
					min_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				}
				min_distance_raw = results[i].m_sumAnswer - epsilon;
				negativeClassDistance = results[i].m_negativeClassAnswer;
				break;
			}
			continue;
		}

		/*if (std::get<0>(results[i]) == 0)
			break;*/

		if (negativeClassValue != results[i].m_target) //if classify != target
		{
			if (min_distance == 1000000.0)
			{
				min_distance = -(results[i].m_sumAnswer / results[0].m_sumAnswer);
				min_distance_raw = -1000000.0; //remove empty space with no vectors
				negativeClassDistance = -1000000.0;
			}
			break;
		}
		if (results[i].m_sumAnswer != 0) //do not accept zero as valid answer because that means that example is so far away that we know nothing
		{
			correctNumber++;
			negativeCorrectNumber++;
			negativeIndex = i;
		}

		if (results[i].m_sumAnswer < 0)
		{
			min_distance = -((results[i].m_sumAnswer / results[0].m_sumAnswer) + (results[i + 1].m_sumAnswer / results[0].m_sumAnswer)) / 2;
		}
		else //if threshold is bigger than 0
		{
			min_distance = ((results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer) + (results[i + 1].m_sumAnswer / results[results.size() - 1].m_sumAnswer)) / 2;
		}
		min_distance_raw = (results[i].m_sumAnswer + results[i + 1].m_sumAnswer) / 2;
		negativeClassDistance = results[i].m_negativeClassAnswer;


		//negativeThrSamples.emplace_back(results[i].m_sumAnswer, correctNumber);

		if (m_distanceSet && m_mode == boundryCheck && min_distance >= m_DistanceBestNegative)
		{
			break;
		}
	}

	std::vector<std::pair<double, int>> positiveThrSamples;
	int positiveCorrectNumber = 0;
	auto positiveIndex = results.size() - 1;
	for (auto i = results.size() - 1; i >= 0; --i)
	{
		if (results[i].m_sumAnswer < hard_coded_threshold_pos)
		{
			if (max_distance == -1000000)
			{
				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				max_distance_raw = 1000000;  //remove empty space with no vectors
				positiveClassDistance = 1000000;
			}
			break;
		}

		
		if (m_useBias && results[i].m_sumAnswer < -bias)
		{
			if (max_distance == -1000000)
			{
				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				max_distance_raw = 1000000;  //remove empty space with no vectors
				positiveClassDistance = 1000000;
			}
			else if (i < results.size() - 2)
			{
				max_distance_raw = (results[i + 1].m_sumAnswer + results[i + 2].m_sumAnswer) / 2;
				positiveClassDistance = results[i + 1].m_positiveClassAnswer;
			}
			else if (i == results.size() - 1)
			{
				max_distance_raw = results[i].m_sumAnswer;
				positiveClassDistance = results[i].m_positiveClassAnswer;
			}
			break;
		}


		//2 following vectors need to have different decision values in order to differentiate them
		if (i != results.size() - 1 && results[i].m_sumAnswer == results[i + 1].m_sumAnswer)
		{
			if (positiveClassValue != results[i].m_target)
			{
				if (results[i].m_sumAnswer > 0)
				{
					max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				}
				else
				{
					max_distance = -results[i].m_sumAnswer / results[0].m_sumAnswer;
				}
				max_distance_raw = results[i].m_sumAnswer + epsilon;
				positiveClassDistance = results[i].m_positiveClassAnswer;
				break;
			}
			continue;
		}


		if (positiveClassValue != results[i].m_target) //if classify != target
		{
			if (max_distance == -1000000)
			{
				max_distance = results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer;
				max_distance_raw = 1000000;  //remove empty space with no vectors
				positiveClassDistance = 1000000;
			}
			break;
		}

		if (results[i].m_sumAnswer != min_distance_raw && results[i].m_sumAnswer != 0)
		{
			correctNumber++;
			positiveCorrectNumber++;
			positiveIndex = i;
		}

		if (results[i].m_sumAnswer > 0)
		{
			max_distance = ((results[i].m_sumAnswer / results[results.size() - 1].m_sumAnswer) + results[i - 1].m_sumAnswer / results[results.size() - 1].m_sumAnswer) / 2;
		}
		else
		{
			max_distance = -(results[i].m_sumAnswer / results[0].m_sumAnswer + results[i - 1].m_sumAnswer / results[0].m_sumAnswer) / 2;
		}
		max_distance_raw = (results[i].m_sumAnswer + results[i - 1].m_sumAnswer) / 2;
		positiveClassDistance = results[i].m_positiveClassAnswer;

		//positiveThrSamples.emplace_back(results[i].m_sumAnswer, positiveCorrectNumber);

		if (max_distance_raw == min_distance_raw && correctNumber == samples.size())
		{
			max_distance = min_distance;
			//min_distance_raw = max_distance_raw = 0;
			break;
		}

		if (m_distanceSet && m_mode == boundryCheck && max_distance <= m_DistanceBestPositive)
		{
			break;
		}
	}

	//std::ofstream debugThrehsolds("D:\\ENSEMBLE_721_BIG_SETSdebug\\" + timeUtils::getTimestamp() + "_thresholds.txt");
	//debugThrehsolds << "Current neg, current pos\n";
	//debugThrehsolds << min_distance_raw << ", " << max_distance_raw << "\n";
	//debugThrehsolds << "Min and max\n";
	//debugThrehsolds << results[0].m_sumAnswer << ", "  << results[results.size() - 1].m_sumAnswer << "\n";
	//debugThrehsolds << "#neg ans, pos ans,  sum, target\n";
	//
	//std::vector<std::pair<double, double>> distanceAndSamples;
	//for(auto& sample : results)
	//{
	//	debugThrehsolds << sample.m_negativeClassAnswer << ", " << sample.m_positiveClassAnswer << ", " << sample.m_sumAnswer << ", " << sample.m_target << "\n";
	//}
	
	//for(auto negativeThr : negativeThrSamples)
	//{
	//	for(auto positiveThr : positiveThrSamples)
	//	{
	//		auto distance = positiveThr.first - negativeThr.first;
	//		auto samplesCovered = static_cast<double>(negativeThr.second + positiveThr.second) / static_cast<double>(samples.size());
	//		//distanceAndSamples.e
	//		debugThrehsolds << negativeThr.first << ", " << positiveThr.first << ", " << distance << ", " << samplesCovered << "\n";
	//	}
	//}


	auto correctlyClassifiedCertainPercent = 10.00 * (static_cast<float>(correctNumber) / static_cast<float>(samples.size()));

	//TODO fix by config
	//auto thresholdNumberOfVectors = 100u;

	//std::vector<double> mcc_scores;
	//if (correctNumber < thresholdNumberOfVectors)
	//{
	//	//set thresholds for max MCC that takes 100 vectors
	//	auto missingCount = thresholdNumberOfVectors - correctNumber;

	//	auto tp = static_cast<unsigned int>(results.size() - positiveIndex);
	//	auto tn = negativeIndex;
	//	auto fp = 0;
	//	auto fn = 0;

	//	for (auto j = positiveIndex - 1; j > positiveIndex - missingCount; j--)
	//	{
	//		if (results[j].m_target == 1)
	//		{
	//			tp++;
	//		}
	//		else
	//		{
	//			fn++;
	//		}
	//	}

	//	for(auto i = negativeIndex + 1; i < negativeIndex + missingCount; i++)
	//	{
	//		if (results[i].m_target == 0)
	//		{
	//			tn++;
	//		}
	//		else
	//		{
	//			fp++;
	//		}

	//		ConfusionMatrix matrix(tp, tn, fp, fn);
	//		auto mcc = matrix.MCC();
	//		mcc_scores.emplace_back(mcc);

	//		auto posIndexCorresponding = positiveIndex - missingCount + 1 + (i - negativeIndex);

	//		if(results[posIndexCorresponding].m_target == 1)
	//		{
	//			tp--;
	//		}
	//		else
	//		{
	//			fn--;
	//		}
	//	}
	//	
	//}

	
	/*auto correctlyClassifiedCertainPercent = static_cast<float>(positiveCorrectNumber) / static_cast<float>(positiveCount)
											+ static_cast<float>(negativeCorrectNumber) / static_cast<float>(negativeCount);*/

	//LOG_F(INFO, "Correctly classified percent %f", correctlyClassifiedCertainPercent);
	//LOG_F(INFO, "correctlyClassifiedCertainPercent %f", correctlyClassifiedCertainPercent);

	epsilon = 0.000001;
	if (std::fabs(max_distance_raw - min_distance_raw) < epsilon && correctlyClassifiedCertainPercent != 1.0)
	{
		//correctlyClassifiedCertainPercent = 0.0;
		//fix error with float epsilon by moving slightly the boundaries otherwise depending on error stacking visualization and classification can be wrong
		max_distance_raw += epsilon;
		min_distance_raw -= epsilon;
		max_distance += epsilon;
		min_distance -= epsilon;
	}

	if (std::isnan(max_distance))
	{
		LOG_F(WARNING, "Max distance was nan, fixing it to 0");
		max_distance = 0;
	}
	if (std::isnan(min_distance))
	{
		LOG_F(WARNING, "Min distance was nan, fixing it to 0");
		min_distance = 0;
	}

	//auto distanceFitness = 1 - ((max_distance - min_distance) / 2);

	setThresholds(svmModel, max_distance_raw, min_distance_raw, max_distance, min_distance);


	//if these threshold are not set they are not used and default value of -1111 is set for them
	if (m_useSingleClassPrediction)
	{
		auto res = reinterpret_cast<phd::svm::SvmLibImplementation*>(svmModel.get());
		res->setClassCertaintyThreshold(negativeClassDistance, positiveClassDistance, negativeClassDistance, positiveClassDistance);
	}
	
	/*auto result = calculateMetric(individual, testSamples, false);
	if (result.m_fitness > 0.2)
	{
		std::cout << "Check what is happening\n";
	}*/
}

#pragma optimize("", on)
} // namespace svmComponents
