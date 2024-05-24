#include "SequentialGammaWorkflowWithFeatureSelection.h"
#include "GridSearchWorkflow.h"
#include "libSvmComponents/RbfKernel.h"
#include "LibGeneticComponents/LocalGlobalAdaptationSelection.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "libSvmComponents/CustomKernelTraining.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "libPlatform/loguru.hpp"

namespace genetic
{
SequentialGammaWorkflowWithFeatureSelection::SequentialGammaWorkflowWithFeatureSelection(const SvmWokrflowConfiguration& config,
                                                                                         SequentialGammaConfigWithFeatureSelection algorithmConfig,
                                                                                         IDatasetLoader& workflow)
	: m_config(config)
	, m_loadingWorkflow(workflow)
	, m_algorithmConfig(std::move(algorithmConfig))
	, m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
	, m_valdiationElement(m_algorithmConfig.m_validationMethod)
	, m_valdiationTestDataElement(
		std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelFeaturesSelectionChromosome>>(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true))
	, m_stopConditionElement(*m_algorithmConfig.m_stopCondition)
	, m_crossoverElement(*m_algorithmConfig.m_crossover)
	, m_mutationElement(*m_algorithmConfig.m_mutation)
	, m_selectionElement(*m_algorithmConfig.m_selection)
	, m_createPopulationElement(*m_algorithmConfig.m_populationGeneration)
	, m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
	, m_educationElement(m_algorithmConfig.m_educationElement)
	, m_crossoverCompensationElement(std::move(m_algorithmConfig.m_crossoverCompensationElement))
	, m_adaptationElement(m_algorithmConfig.m_adaptationElement)
	, m_superIndividualsGenerationElement(m_algorithmConfig.m_superIndividualsGenerationElement)
	, m_trainingSuperIndividualsElement(*m_algorithmConfig.m_training)
	, m_validationSuperIndividualsElement(m_algorithmConfig.m_validationMethod)
	, m_parentSelectionElement(*m_algorithmConfig.m_parentSelection)
	, m_compensationGenerationElement(std::move(m_algorithmConfig.m_compensationGenerationElement))
	, m_trainingSet(nullptr)
	, m_validationSet(nullptr)
	, m_testSet(nullptr)
	, m_generationNumber(0)
	, m_numberOfClassExamples(m_algorithmConfig.m_numberOfClassExamples)
	, m_initialNumberOfClassExamples(m_algorithmConfig.m_numberOfClassExamples)
	, m_featureSetOptimization(m_algorithmConfig.m_featureSetOptimization)
	, m_useFeatureOptimization(false)
{
	m_shrinkTrainingSet = true;
	m_buildEnsamble = false;
}

//TODO Unused can be removed in future
class FilteredDatasetLoader : public IDatasetLoader
{
public:
	FilteredDatasetLoader(const FeatureSetOptimizationWorkflow featureSetOptimization)
		: m_featureSetOptimization(featureSetOptimization)
	{}
	
	virtual const dataset::Dataset<std::vector<float>, float>& getTraningSet()
	{
		m_tr = m_featureSetOptimization->getFilteredTraningSet();
		return m_tr;
	}
	virtual const dataset::Dataset<std::vector<float>, float>& getValidationSet()
	{
		m_val = m_featureSetOptimization->getFilteredValidationSet();
		return m_val;
	}
	virtual const dataset::Dataset<std::vector<float>, float>& getTestSet()
	{
		m_test = m_featureSetOptimization->getFilteredTestSet();
		return m_test;
	}
	bool isDataLoaded() const override { return true; }
	const std::vector<float>& scalingVectorMin() override { throw std::runtime_error("Not implemented scalingVectorMin SequentialGammaWorkflowWithFeatureSelection"); }
	const std::vector<float>& scalingVectorMax() override { throw std::runtime_error("Not implemented scalingVectorMax SequentialGammaWorkflowWithFeatureSelection"); }

private:
	const FeatureSetOptimizationWorkflow m_featureSetOptimization;

	dataset::Dataset<std::vector<float>, float> m_tr;
	dataset::Dataset<std::vector<float>, float> m_val;
	dataset::Dataset<std::vector<float>, float> m_test;
};

std::shared_ptr<phd::svm::ISvm> SequentialGammaWorkflowWithFeatureSelection::run()
{
	initializeGeneticAlgorithm();
	runGeneticAlgorithm();
	m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));

	return m_population.getBestOne().getClassifier();
}

void SequentialGammaWorkflowWithFeatureSelection::initializeGeneticAlgorithm()
{
	if (m_trainingSet == nullptr)
	{
		m_trainingSet = &m_loadingWorkflow.getTraningSet();
		m_validationSet = &m_loadingWorkflow.getValidationSet();
		m_testSet = &m_loadingWorkflow.getTestSet();
	}

	//feature selection here
	m_featureSetOptimization->initialize();
	//auto filteredDatasetLoader = FilteredDatasetLoader(m_featureSetOptimization);
	

	GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	                                                                       1,
	                                                                       1,
	                                                                       m_numberOfClassExamples,
	                                                                       1,
	                                                                       std::make_shared<svmComponents::SvmKernelTraining>(
		                                                                       m_algorithmConfig.m_svmConfig,
		                                                                       m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::Auc),
	                                                                       std::make_shared<svmComponents::RbfKernel>(svmComponents::ParamGrid(0.001, 1050, 10),
	                                                                                                                  svmComponents::ParamGrid(0.001, 1050, 10),
	                                                                                                                  false)),
							m_loadingWorkflow);

	auto bestOne = gs.run();

	//pnly work for subset validation
	//m_valdiationElement->generateNewSubset(*m_validationSet);
	//m_validationSuperIndividualsElement->generateNewSubset(*m_validationSet);

	try
	{
		m_gammaRange = {bestOne->getGamma() / 10, bestOne->getGamma(), bestOne->getGamma() * 10, bestOne->getGamma() * 100, bestOne->getGamma() * 1000};
		m_CValue = bestOne->getC();

		
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
	}
}

void SequentialGammaWorkflowWithFeatureSelection::internalVisualization_Debugging(
	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> /*pop*/)
{
	//int i = 0;
	//std::ofstream osobniki (m_config.outputFolderPath.string() + "\\details\\gen__" + std::to_string(m_generationNumber) + ".txt");
	//for (const auto& individual : pop)
	//{
	//	//if (individual.getFitness() > 0.6)
	//	{
	//		filesystem::FileSystem fs;
	//		auto detailsPath = m_config.outputFolderPath.string() + "\\details\\";
	//		fs.createDirectories(detailsPath);
	//		auto out = std::filesystem::path(detailsPath + getTimestamp() +
	//			"id__" + std::to_string(i) + ".png");

	//		auto svm = individual.getClassifier();
	//		svmComponents::SvmVisualization visualization;
	//		auto m_image = visualization.createDetailedVisualization(*svm,
	//			m_algorithmConfig.m_svmConfig.m_height,
	//			m_algorithmConfig.m_svmConfig.m_width,
	//			*m_trainingSet, *m_validationSet);
	//		//auto img =  gsl::make_span(m_image);

	//		m_savePngElement.launch(m_image, out);
	//		++i;

	//		osobniki << "id: " << i << "\n";

	//		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
	//		osobniki << "Tr: ";
	//		for (auto g : individual.getDataset())
	//		{
	//			osobniki << g.id << ", ";
	//		}
	//		for (auto v : m_frozenSV_ids)
	//		{
	//			osobniki << v.id << ", ";
	//		}
	//		osobniki << std::endl;

	//		osobniki << "Gammas: ";
	//		for (auto j = 0; j < res->m_model->l; ++j)
	//			osobniki << res->m_model->param.gammas_after_training->at(j) << " ";
	//		osobniki << std::endl;

	//		osobniki << "Alphas: ";
	//		for (auto j = 0; j < res->m_model->l; ++j)
	//		{
	//			osobniki << res->m_model->sv_coef[0][j] << " ";
	//		}
	//		osobniki << "Training chromosome: ";
	//		for (auto j = 0; j < individual.getDataset().size(); ++j)
	//		{
	//			osobniki << individual.getDataset()[j].gamma << ":" << individual.getDataset()[j].classValue << "  ";
	//		}
	//		osobniki << "\n";
	//	}
	//}
	//osobniki.close();
}

struct auc2Results
{
	auc2Results(svmComponents::ConfusionMatrix matrix, double auc2Value, double optimalThreshold)
		: m_matrix(std::move(matrix))
		, m_auc2Value(auc2Value)
		, m_optimalThreshold(optimalThreshold)
	{
	}

	const svmComponents::ConfusionMatrix m_matrix;
	const double m_auc2Value;
	const double m_optimalThreshold;
};

double trapezoidArea2(double x1, double x2, double y1, double y2)
{
	auto base = std::fabs(x1 - x2);
	auto height = (y1 + y2) / 2.0;
	return height * base;
}

auc2Results auc2(std::vector<std::pair<double, int>>& probabilityTargetPair, int negativeCount, int positiveCount)
{
	std::sort(probabilityTargetPair.begin(), probabilityTargetPair.end(), [](const auto& a, const auto& b)
	{
		return a.first > b.first;
	});

	double auc2 = 0;
	double previousProbability = -1;
	int falsePositive = 0;
	int truePositive = 0;
	int falsePositivePreviousIteration = 0;
	int truePositivesPreviousIteration = 0;

	auto maxAccuracyForThreshold = 0.0;
	auto threshold = 0.0;
	svmComponents::ConfusionMatrix matrixWithOptimalThreshold(0u, 0u, 0u, 0u);

	for (const auto& pair : probabilityTargetPair)
	{
		auto [probability, label] = pair;

		if (probability != previousProbability)
		{
			auc2 += trapezoidArea2(falsePositive, falsePositivePreviousIteration, truePositive, truePositivesPreviousIteration);
			previousProbability = probability;
			falsePositivePreviousIteration = falsePositive;
			truePositivesPreviousIteration = truePositive;
		}

		label == 1 ? truePositive++ : falsePositive++;

		const auto accuracyForThreshold = static_cast<double>(truePositive + (negativeCount - falsePositive)) / static_cast<double>(negativeCount +
			positiveCount);
		if (accuracyForThreshold > maxAccuracyForThreshold)
		{
			matrixWithOptimalThreshold = svmComponents::ConfusionMatrix(truePositive, (negativeCount - falsePositive),
			                                                            falsePositive, (positiveCount - truePositive));
			maxAccuracyForThreshold = accuracyForThreshold;
			threshold = probability;
		}
	}
	auc2 += trapezoidArea2(negativeCount, falsePositivePreviousIteration, positiveCount, truePositivesPreviousIteration);
	auc2 /= (static_cast<double>(positiveCount) * static_cast<double>(negativeCount));

	return auc2Results(matrixWithOptimalThreshold, auc2, threshold);
}

svmComponents::Metric ensembleLastEpoch2(const dataset::Dataset<std::vector<float>, float>& data,
                                        geneticComponents::Population<svmComponents::SvmCustomKernelChromosome>& population)
{
	auto samples = data.getSamples();
	auto targets = data.getLabels();

	std::vector<float> hyperplaneDistancePerSample;
	hyperplaneDistancePerSample.resize(samples.size());

	for (auto& individual : population)
	{
		auto classifier = individual.getClassifier();
		for (auto f = 0; f < samples.size(); ++f)
		{
			hyperplaneDistancePerSample[f] += classifier->classifyHyperplaneDistance(samples[f]);
		}
	}

	auto positiveCount = static_cast<unsigned int>(std::count_if(targets.begin(), targets.end(),
	                                                             [](const auto& target)
	                                                             {
		                                                             return target == 1;
	                                                             }));
	auto negativeCount = static_cast<unsigned int>(samples.size() - positiveCount);

	std::vector<std::pair<double, int>> probabilites;
	probabilites.reserve(targets.size());

	for (auto i = 0u; i < targets.size(); i++)
	{
		probabilites.emplace_back(std::make_pair(hyperplaneDistancePerSample[i], static_cast<int>(targets[i])));
	}
	auc2Results result = auc2(probabilites, negativeCount, positiveCount);
	//svmModel->setOptimalProbabilityThreshold(result.m_optimalThreshold);  //TODO think of other way of setting this
	return svmComponents::Metric(result.m_auc2Value, result.m_matrix);
}

void SequentialGammaWorkflowWithFeatureSelection::visualizeFrozenSet(
	geneticComponents::Population<SequentialGammaWorkflowWithFeatureSelection::SvmCustomKernelChromosome>& best_pop)
{
	auto copy = m_frozenSV;

	auto svm = m_population.getBestOne().getClassifier();
	svmComponents::SvmVisualization visualization;

	auto best = m_population.getBestOne();

	svmComponents::SvmVisualization visualization3;
	visualization3.setGene(best);
	//auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
	//auto[map, scores] = res2->check_sv(*m_trainingSet);
	//visualization3.setScores(scores);
	//visualization3.setMap(map);

	auto image3 = visualization3.createDetailedVisualization(*best_pop.getBestOne().getClassifier(), 500, 500, *m_trainingSet, *m_trainingSet);
	SvmWokrflowConfiguration config_copy3{"", "", "", m_config.outputFolderPath, "m_supportVectorFrozenPool", ""};
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, m_generationNumber);
	m_savePngElement.launch(image3, m_pngNameSource);
}

bool SequentialGammaWorkflowWithFeatureSelection::shrinkTrainingSet(
	geneticComponents::Population<SequentialGammaWorkflowWithFeatureSelection::SvmCustomKernelChromosome>& best_pop)
{
	dataset::Dataset<std::vector<float>, float> new_Training;
	std::unordered_set<uint64_t> new_training_set_ids;
	auto samples = m_trainingSet->getSamples();
	auto labels = m_trainingSet->getLabels();
	//auto classifier = m_population.getBestOne().getClassifier();
	//auto classifier = best_pop.getBestOne().getClassifier();  //m_population.getBestOne().getClassifier();

	if (m_algorithmConfig.m_shrinkOnBestOnly)
	{
		auto classifier = m_population.getBestOne().getClassifier();
		for (auto f = 0; f < samples.size(); ++f)
		{
			if (classifier->classifyWithOptimalThreshold(samples[f]) == labels[f])
			{
				continue;
			}
			else
			{
				new_training_set_ids.emplace(f);
				new_Training.addSample(samples[f], labels[f]);
			}
		}
	}
	else if (!m_algorithmConfig.m_trainAlpha)
	{
		auto classifier = m_population.getBestOne().getClassifier();
		for (auto f = 0; f < samples.size(); ++f)
		{
			if (classifier->classifyWithOptimalThreshold(samples[f]) == labels[f])
			{
				continue;
			}
			else
			{
				new_training_set_ids.emplace(f);
				new_Training.addSample(samples[f], labels[f]);
			}
		}
	}
	else
	{
		for (auto& individual : m_population)
		{
			auto classifier = individual.getClassifier();
			for (auto f = 0; f < samples.size(); ++f)
			{
				if (classifier->classifyWithOptimalThreshold(samples[f]) == labels[f])
					//if (classifier->classify(samples[f]) == labels[f])
				{
					continue;
				}
				else
				{
					new_training_set_ids.emplace(f);
					new_Training.addSample(samples[f], labels[f]);
				}
			}
		}
	}

	std::unordered_set<uint64_t> forbidden_set;
	for (auto j = 0; j < samples.size(); ++j)
	{
		if (new_training_set_ids.find(j) == new_training_set_ids.end())
		{
			forbidden_set.emplace(j);
		}
	}

	m_forbiddenSetSize = forbidden_set.size();

	if (forbidden_set.size() == m_trainingSet->size())
		return true;

	if (m_algorithmConfig.m_svmConfig.m_doVisualization)
	{
		svmComponents::SvmVisualization visualization2;
		auto image2 = visualization2.createVisualizationNewTrainingSet(*best_pop.getBestOne().getClassifier(), 500, 500, new_Training);
		SvmWokrflowConfiguration config_copy2{"", "", "", m_config.outputFolderPath, "new_training_set_", ""};
		setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, m_generationNumber);
		m_savePngElement.launch(image2, m_pngNameSource);
	}

	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setForbbidens(forbidden_set);
	reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setForbbidens(forbidden_set);
	m_crossoverCompensationElement.setForbbidens(forbidden_set);

	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setImbalancedOrOneClass(true);
	m_crossoverCompensationElement.setImbalancedOrOneClass(true);
	reinterpret_cast<const std::shared_ptr<svmComponents::CrossoverCustomGauss>&>(m_algorithmConfig.m_crossover)->setImbalancedOrOneClass(true);
	m_superIndividualsGenerationElement->setImbalancedOrOneClass(true);
	return false;
}

void SequentialGammaWorkflowWithFeatureSelection::shrinkValidationSet()
{
	auto samples = m_validationSet->getSamples();
	auto labels = m_validationSet->getLabels();
	dataset::Dataset<std::vector<float>, float> new_validation;
	std::unordered_set<uint64_t> new_validation_set_ids;

	for (auto& individual : m_population)
	{
		auto classifier = individual.getClassifier();
		for (auto f = 0; f < samples.size(); ++f)
		{
			if (classifier->classifyWithOptimalThreshold(samples[f]) != labels[f] && new_validation_set_ids.emplace(f).second)
			{
				new_validation.addSample(samples[f], labels[f]);
			}
		}
	}

	std::cout << "Shrink coefficient: " << new_validation.size() / static_cast<float>(m_validationSet->size()) << "\n";
	std::cout << "New validation size: " << new_validation.size() << "\n";

	auto classCount = svmComponents::svmUtils::countLabels(2, new_validation);

	if (classCount[0] != 0 && classCount[1] != 0)
	{
		m_validationSet2 = new_validation;
		m_validationSet = &m_validationSet2;
	}
}

void SequentialGammaWorkflowWithFeatureSelection::buildEnsembleFromLastGeneration()
{
	auto svNumber = 0;
	for (auto& p : m_population)
	{
		svNumber += p.getNumberOfSupportVectors();
	}

	auto validationResults = ensembleLastEpoch2(*m_validationSet, m_population);
	auto testResults = ensembleLastEpoch2(*m_testSet, m_population);

	auto validationDataset = *m_validationSet;
	auto featureNumber = validationDataset.getSamples()[0].size();

	std::string logInfo;
	logInfo += m_algorithmName;
	logInfo += "\t";
	logInfo += std::to_string(m_generationNumber).append("\t");
	logInfo += std::to_string(m_population.size()).append("\t");
	logInfo += std::to_string(m_timer.getTimeMiliseconds().count()).append("\t");
	logInfo += std::to_string(validationResults.m_fitness).append("\t");
	logInfo += std::to_string(validationResults.m_fitness).append("\t");
	logInfo += std::to_string(svNumber).append("\t");
	logInfo += std::to_string(svNumber).append("\t");
	logInfo += std::to_string(testResults.m_fitness).append("\t");
	logInfo += std::to_string(testResults.m_fitness).append("\t");
	logKernelParameters(m_population.getBestOne(), logInfo);
	logInfo += std::to_string(1).append("\t");
	logInfo += std::to_string(0.0).append("\t");
	logInfo += std::to_string(0.0).append("\t");
	logInfo += std::to_string(0.0).append("\t");
	logInfo += std::to_string(0.0).append("\t");
	logInfo += std::to_string(svmComponents::Accuracy(validationResults.m_confusionMatrix.value())).append("\t");
	logInfo += std::to_string(featureNumber).append("\t");
	logInfo += std::to_string(m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size() + m_frozenSV.size()).append("\t");
	logInfo += validationResults.m_confusionMatrix.value().to_string().append("\t");
	logInfo += testResults.m_confusionMatrix.value().to_string().append("\t");

	logInfo.append("\n");

	m_resultLogger.customLogEntry(logInfo);

	//int h = 0;
	//for (auto& p : m_population)
	//{
	//	SvmWokrflowConfiguration config_copy2{ "","","",m_config.outputFolderPath, "ensemble_part" + std::to_string(h),"" };
	//	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, m_generationNumber);
	//	//setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
	//	h++;
	//	auto svm = p.getClassifier();
	//	svmComponents::SvmVisualization visualization;		
	//	auto best = p;
	//	visualization.setGene(best);
	//	auto image = visualization.createVisualization(*svm,
	//		m_algorithmConfig.m_svmConfig.m_height,
	//		m_algorithmConfig.m_svmConfig.m_width,
	//		*m_trainingSet, *m_validationSet);

	//	m_savePngElement.launch(image, m_pngNameSource);
	//}
}

void SequentialGammaWorkflowWithFeatureSelection::runGeneticAlgorithm()
{
	try
	{
		auto previousIterationFitness = 0.0;
		geneticComponents::Population<svmComponents::SvmCustomKernelFeaturesSelectionChromosome> popBackup;
		geneticComponents::Population<svmComponents::SvmCustomKernelFeaturesSelectionChromosome> popBackupTestScore;

		//for given gamma range
		for (auto i = 0u; i < m_gammaRange.size(); ++i)
		{
			m_currentGamma = m_gammaRange[i];

			reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
					setNumberOfClassExamples(m_numberOfClassExamples);
			m_adaptationElement.resetToInitial(m_numberOfClassExamples);

			m_superIndividualsGenerationElement->setC(m_CValue);
			reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
					setCandGammaSingle(m_CValue, m_gammaRange[i]);
			reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
			m_crossoverCompensationElement.setGamma(m_gammaRange[i]);
			m_supportVectorPoolElement.setCurrentGamma(m_gammaRange[i]);

			double initialBest = 0.0;
			if (!m_populationWithFeatures.empty())
			{
				initialBest = m_populationWithFeatures.getBestOne().getFitness();
			}

			initMemetic();

			if (m_populationWithFeatures.getBestOne().getFitness() - initialBest < 0.0)
			{
				m_populationWithFeatures = popBackup;
				logResults(m_populationWithFeatures, popBackupTestScore);

				continue;
			}

			memeticAlgorithm();

			if (m_populationWithFeatures.getBestOne().getFitness() - previousIterationFitness < 0.0)
			{
				m_populationWithFeatures = popBackup;
				logResults(m_populationWithFeatures, popBackupTestScore);
				continue;
			}

			previousIterationFitness = m_populationWithFeatures.getBestOne().getFitness();
			//sort of early stopping
			if (m_populationWithFeatures.getBestOne().getFitness() == 1.0)
				break;

			
			auto bestVectors = m_population.getBestOne().getDataset();

			for (auto& g : bestVectors)
			{
				m_frozenSV_ids.insert(g);
			}
			m_frozenSV.clear();
			m_frozenSV.insert(std::end(m_frozenSV), std::begin(m_frozenSV_ids), std::end(m_frozenSV_ids));

			//visualize the best vectors - train the best one and plot the results
			
			geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome> best_pop;
			auto copy2 = m_frozenSV;
			svmComponents::SvmCustomKernelChromosome best_vec{std::move(copy2), m_population.getBestOne().getC()};

			auto f = getFeaturePop().getBestOne();

			auto temp = SvmCustomKernelFeaturesSelectionChromosome(best_vec, f);
			best_pop = geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>{ std::vector<SvmCustomKernelFeaturesSelectionChromosome>{ temp } };
			m_trainingSvmClassifierElement.launch(best_pop, *m_trainingSet);

			auto bestCustomKernel = getCustomKernelPopFromMerged(best_pop);
			
			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				visualizeFrozenSet(bestCustomKernel);
			}

			//shrink training set
			if (m_shrinkTrainingSet)
			{
				if (shrinkTrainingSet(bestCustomKernel))
					break;
			}

			//set parameters and reset MASVM state
			reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
					setCandGammaSingle(m_CValue, m_gammaRange[i]);
			reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
			m_crossoverCompensationElement.setGamma(m_gammaRange[i]);


			//TODO
			popBackup = m_populationWithFeatures;
			auto copy = m_populationWithFeatures; // mergePopulations(m_population);
			
			auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);
			popBackupTestScore = testPopulation;
		}

		m_useFeatureOptimization = true;

		//TODO take best individual and save in here for later


		auto savedBestOne = m_populationWithFeatures.getBestOne();






		for (auto i = m_gammaRange.size() - 1; i < m_gammaRange.size(); ++i)
		{
			m_currentGamma = m_gammaRange[i];

			reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
				setNumberOfClassExamples(m_numberOfClassExamples);
			m_adaptationElement.resetToInitial(m_numberOfClassExamples);

			m_superIndividualsGenerationElement->setC(m_CValue);
			reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
				setCandGammaSingle(m_CValue, m_gammaRange[i]);
			reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
			m_crossoverCompensationElement.setGamma(m_gammaRange[i]);
			m_supportVectorPoolElement.setCurrentGamma(m_gammaRange[i]);

			double initialBest = 0.0;
			if (!m_populationWithFeatures.empty())
			{
				initialBest = m_populationWithFeatures.getBestOne().getFitness();
			}

			initMemetic();

			if (m_populationWithFeatures.getBestOne().getFitness() - initialBest < 0.0)
			{
				m_populationWithFeatures = popBackup;
				logResults(m_populationWithFeatures, popBackupTestScore);

				continue;
			}

			memeticAlgorithm();

			if (m_populationWithFeatures.getBestOne().getFitness() - previousIterationFitness < 0.0)
			{
				m_populationWithFeatures = popBackup;
				logResults(m_populationWithFeatures, popBackupTestScore);
				continue;
			}

			previousIterationFitness = m_populationWithFeatures.getBestOne().getFitness();
			//sort of early stopping
			if (m_populationWithFeatures.getBestOne().getFitness() == 1.0)
				break;


			auto bestVectors = m_population.getBestOne().getDataset();

			for (auto& g : bestVectors)
			{
				m_frozenSV_ids.insert(g);
			}
			m_frozenSV.clear();
			m_frozenSV.insert(std::end(m_frozenSV), std::begin(m_frozenSV_ids), std::end(m_frozenSV_ids));

			//visualize the best vectors - train the best one and plot the results

			geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome> best_pop;
			auto copy2 = m_frozenSV;
			svmComponents::SvmCustomKernelChromosome best_vec{ std::move(copy2), m_population.getBestOne().getC() };

			auto f = getFeaturePop().getBestOne();

			auto temp = SvmCustomKernelFeaturesSelectionChromosome(best_vec, f);
			best_pop = geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>{ std::vector<SvmCustomKernelFeaturesSelectionChromosome>{ temp } };
			m_trainingSvmClassifierElement.launch(best_pop, *m_trainingSet);

			auto bestCustomKernel = getCustomKernelPopFromMerged(best_pop);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				visualizeFrozenSet(bestCustomKernel);
			}

			//shrink training set
			if (m_shrinkTrainingSet)
			{
				if (shrinkTrainingSet(bestCustomKernel))
					break;
			}

			//set parameters and reset MASVM state
			reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->
				setCandGammaSingle(m_CValue, m_gammaRange[i]);
			reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
			m_crossoverCompensationElement.setGamma(m_gammaRange[i]);


			//TODO
			popBackup = m_populationWithFeatures;
			auto copy = m_populationWithFeatures; // mergePopulations(m_population);

			auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);
			popBackupTestScore = testPopulation;
		}




		if(m_populationWithFeatures.getBestOne().getFitness() > savedBestOne.getFitness())
		{
			std::cout << "Feature Selection improved something\n\n !!!";
			LOG_F(INFO, "Feature Selection improved something");
		}


		//log proper results
		//return proper SVM


		if (m_buildEnsamble)
		{
			buildEnsembleFromLastGeneration();
		}
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
	}
}

void SequentialGammaWorkflowWithFeatureSelection::logResults(const geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>& population,
                                                             const geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>& testPopulation)
{
	auto bestOneConfustionMatrix = population.getBestOne().getConfusionMatrix().value();
	auto validationDataset = *m_validationSet;
	auto featureNumber = population.getBestOne().featureSetSize(); //validationDataset.getSamples()[0].size();
	auto bestOneIndex = population.getBestIndividualIndex();

	auto reductionPercent = static_cast<double>(m_forbiddenSetSize) / static_cast<double>(m_trainingSet->size());

	m_resultLogger.createLogEntry(population,
	                              testPopulation,
	                              m_timer,
	                              m_algorithmName,
	                              m_generationNumber,
	                              svmComponents::Accuracy(bestOneConfustionMatrix),
	                              featureNumber,
	                              m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size() + m_frozenSV.size(),
	                              bestOneConfustionMatrix,
	                              testPopulation[bestOneIndex].getConfusionMatrix().value(),
	                              reductionPercent);
	m_generationNumber++;
}

void SequentialGammaWorkflowWithFeatureSelection::initMemetic()
{
	try
	{
		if (m_generationNumber % m_algorithmConfig.m_genererateEveryGeneration == 0)
		{
			//pnly work for subset validation
			 m_valdiationElement->generateNewSubset(*m_validationSet);
			m_validationSuperIndividualsElement->generateNewSubset(*m_validationSet);
		}

		m_svPool.clear();
		m_supportVectorPoolElement.clear();
		//m_numberOfClassExamples = m_initialNumberOfClassExamples;
		//m_adaptationElement.resetToInitial(m_initialNumberOfClassExamples);
		m_adaptationElement.setFrozenSetSize(static_cast<unsigned int>(m_frozenSV.size()));

		auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);


	
		
		auto tr = reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernelFS>&>(m_algorithmConfig.m_training);
		auto merged = mergePopulations(population);
		tr->trainPopulation(merged, *m_trainingSet, m_frozenSV);

		merged = m_valdiationElement->launch(merged, *m_validationSet);
		m_populationWithFeatures = merged;
		
		for (auto& individual : m_population)
		{
			auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(individual.getClassifier().get());
			if (res2->isMaxIterReached() && m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				svmComponents::SvmVisualization visualization2;
				visualization2.setGene(individual);
				visualization2.setGammasValues(m_gammaRange);
				visualization2.setFrozenSet(m_frozenSV);
				auto image2 = visualization2.createDetailedVisualization(*individual.getClassifier(), 500, 500, *m_trainingSet, *m_trainingSet);
				SvmWokrflowConfiguration config_copy2{"", "", "", m_config.outputFolderPath, "max_iter_Problem", ""};
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, m_generationNumber);
				m_savePngElement.launch(image2, m_pngNameSource);
			}
		}
		internalVisualization_Debugging(m_population);

		//TODO
		auto copy = merged;
		auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);

			auto svm = m_population.getBestOne().getClassifier();
			svmComponents::SvmVisualization visualization;
			//auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
			auto best = m_population.getBestOne();

			visualization.setGene(best);
			visualization.setGammasValues(m_gammaRange);
			visualization.setFrozenSet(m_frozenSV);

			auto image = visualization.createDetailedVisualization(*svm,
			                                                       m_algorithmConfig.m_svmConfig.m_height,
			                                                       m_algorithmConfig.m_svmConfig.m_width,
			                                                       *m_trainingSet, *m_validationSet);

			m_savePngElement.launch(image, m_pngNameSource);
		}

		//logAllModels(testPopulation);
		logResults(m_populationWithFeatures, testPopulation);
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
		throw;
	}
}

void SequentialGammaWorkflowWithFeatureSelection::memeticAlgorithm()
{
	try
	{
		bool isStop = false;

		while (!isStop)
		{
			if (m_generationNumber % m_algorithmConfig.m_genererateEveryGeneration == 0)
			{
				//pnly work for subset validation
				m_valdiationElement->generateNewSubset(*m_validationSet);
				m_validationSuperIndividualsElement->generateNewSubset(*m_validationSet);
			}

			m_population = getCustomKernelPopFromMerged(m_populationWithFeatures);
			auto parents = m_parentSelectionElement.launch(m_population);
			auto newPopulation = m_crossoverElement.launch(parents);

			auto compensantionInfo = m_compensationGenerationElement.generate(parents, m_numberOfClassExamples);
			auto result = m_crossoverCompensationElement.compensate(newPopulation, compensantionInfo);

			m_educationElement->educatePopulation(result, m_svPool, parents, *m_trainingSet);
			auto populationEducated = result;

			populationEducated = m_mutationElement.launch(populationEducated);

			auto tr = reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernelFS>&>(m_algorithmConfig.m_training);

			
			////TODO make sure to have proper initial feature pop in here
			if (m_useFeatureOptimization)
			{
				// merge feature population only to get proper on from initialize, perform all genetic opps and merge again --- check this !!!!!
				//mergePopulations();
				auto  fPop = getFeaturePopFromMerged(m_populationWithFeatures);
				m_featureSetOptimization->performGeneticOperations(fPop);
			}
			auto mergedPop = mergePopulations(populationEducated);

			tr->trainPopulation(mergedPop, *m_trainingSet, m_frozenSV);
			auto poptrained = mergedPop;
			auto afterValidtion = m_valdiationElement->launch(mergedPop, *m_validationSet);
			
			/*tr->trainPopulation(populationEducated, *m_trainingSet, m_frozenSV);
			auto poptrained = populationEducated;
			auto afterValidtion = m_valdiationElement->launch(populationEducated, *m_validationSet);*/

			//internalVisualization_Debugging(populationEducated);

			auto poptrained2 = getCustomKernelPopFromMerged(mergedPop);
			m_supportVectorPoolElement.updateSupportVectorPool(poptrained2, *m_trainingSet);
			m_svPool = m_supportVectorPoolElement.getSupportVectorPool();

			auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
			auto superIndividualsPopulation = m_superIndividualsGenerationElement->createPopulation(superIndividualsSize, m_svPool, m_numberOfClassExamples);


			//TODO
			auto superIndividualsMerged = mergePopulations(superIndividualsPopulation); //TODO merge with best one only
			tr->trainPopulation(superIndividualsMerged, *m_trainingSet, m_frozenSV);
			m_validationSuperIndividualsElement->launch(superIndividualsMerged, *m_validationSet);

			auto combinedPopulation = m_populationCombinationElement.launch(afterValidtion, superIndividualsMerged);
			m_populationWithFeatures = m_selectionElement.launch(m_populationWithFeatures, combinedPopulation);

			m_population = getCustomKernelPopFromMerged(m_populationWithFeatures);
			
			

			/*auto [IsModeLocal, NumberOfClassExamples] = */
			m_adaptationElement.adapt(m_population);
			auto IsModeLocal = m_adaptationElement.getIsModeLocal();
			auto NumberOfClassExamples = m_adaptationElement.getNumberOfClassExamples();
			m_numberOfClassExamples = NumberOfClassExamples;

			
			
			//mergePopulations();
			auto copy = m_populationWithFeatures;
			auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);

			isStop = m_stopConditionElement.launch(m_populationWithFeatures);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);

				auto svm = m_population.getBestOne().getClassifier();
				svmComponents::SvmVisualization visualization;
				//auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
				auto best = m_population.getBestOne();

				visualization.setGene(best);
				//auto[map, scores] = res2->check_sv(*m_validationSet);
				//visualization.setScores(scores);
				//visualization.setMap(map);
				visualization.setGammasValues(m_gammaRange);
				visualization.setFrozenSet(m_frozenSV);

				auto image = visualization.createDetailedVisualization(*svm,
				                                                       m_algorithmConfig.m_svmConfig.m_height,
				                                                       m_algorithmConfig.m_svmConfig.m_width,
				                                                       *m_trainingSet, *m_validationSet);

				m_savePngElement.launch(image, m_pngNameSource);
			}

			auto localGlobal = dynamic_cast<geneticComponents::LocalGlobalAdaptationSelection<SvmCustomKernelChromosome>*>(m_algorithmConfig
			                                                                                                               .m_parentSelection.get());
			if (localGlobal != nullptr)
			{
				localGlobal->setMode(IsModeLocal);
			}

			//logAllModels(testPopulation);
			//getCustomKernelPopFromMerged();
			//auto testPopulation2 = getCustomKernelPopFromMerged(testPopulation);
			logResults(m_populationWithFeatures, testPopulation);

			if (m_shrinkTrainingSet && !m_algorithmConfig.m_trainAlpha)
			{
				if (shrinkTrainingSet(m_population))
					continue;
			}
		}
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
		throw;
	}
}

void SequentialGammaWorkflowWithFeatureSelection::switchMetric()
{
	/*m_estimationMethod = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::auc2);
	m_valdiationElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod);
	m_valdiationTestDataElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod);
	m_validationSuperIndividualsElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod);*/
}
} // namespace genetic
