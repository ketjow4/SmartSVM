#include "RbfLinearWorkflow.h"
#include "GridSearchWorkflow.h"
#include "libSvmComponents/RbfKernel.h"
#include "LibGeneticComponents/LocalGlobalAdaptationSelection.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "libSvmComponents/CustomKernelTraining.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "libPlatform/loguru.hpp"
#include "libSvmComponents/GaSvmGeneration.h"
#include "libSvmComponents/LinearKernel.h"
#include "libSvmComponents/SvmAucMetric.h"
#include "libSvmComponents/SvmAucprcMetric.h"

namespace genetic
{
RbfLinearWorkflow::RbfLinearWorkflow(const SvmWokrflowConfiguration& config,
                                     svmComponents::RbfLinearConfig algorithmConfig,
                                     IDatasetLoader& workflow)
	: m_config(config)
	, m_loadingWorkflow(workflow)
	, m_algorithmConfig(std::move(algorithmConfig))
	, m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
	, m_valdiationElement(m_algorithmConfig.m_validationMethod)
	, m_valdiationTestDataElement(std::make_shared< svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true))
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
{
	m_shrinkTrainingSet = true;
}

std::shared_ptr<phd::svm::ISvm> RbfLinearWorkflow::run()
{
	initializeGeneticAlgorithm();
	runGeneticAlgorithm();
	m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));
	
	return m_population.getBestOne().getClassifier();
}

void RbfLinearWorkflow::setC(double C)
{
	m_CValue = C;
}

void RbfLinearWorkflow::initializeGeneticAlgorithm()
{
	if (m_trainingSet == nullptr)
	{
		m_trainingSet = &m_loadingWorkflow.getTraningSet();
		m_validationSet = &m_loadingWorkflow.getValidationSet();
		m_testSet = &m_loadingWorkflow.getTestSet();
	}

	//GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	//                                                                       1,
	//                                                                       1,
	//                                                                       m_numberOfClassExamples,
	//                                                                       1,
	//                                                                       std::make_shared<svmComponents::SvmKernelTraining>(
	//	                                                                       m_algorithmConfig.m_svmConfig,
	//	                                                                       m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::
	//	                                                                       Auc),
	//                                                                       std::make_shared<svmComponents::RbfKernel>(cv::ml::ParamGrid(0.001, 1050, 10),
	//                                                                                                                  cv::ml::ParamGrid(0.001, 1050, 10),
	//                                                                                                                  false)),
	//                      m_loadingWorkflow);

	//auto bestOne = gs.run();

	//try
	//{
	//	m_gammaRange = { -1, bestOne->getGamma() / 10, bestOne->getGamma(), bestOne->getGamma() * 10, bestOne->getGamma() * 100, bestOne->getGamma() * 1000};
	//	//m_gammaRange = {-1, 100, 500, 1000};
	//	m_CValue = bestOne->getC();
	//	//m_CValue = 1.0;
	//	std::cout << m_CValue;
	//	//set all parameters in here!!!!
	//	//std::vector<double> gammaTest = {/*0.1,*/50, 100 ,200, 1000 };
	//	//m_gammaRange = gammaTest;
	//	//m_CValue = 10; 
	//}
	//catch (const std::exception& exception)
	//{
	//	LOG_F(ERROR, "Error: %s", exception.what());
	//	std::cout << exception.what();
	//}
}

void RbfLinearWorkflow::internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> /*pop*/)
{
	//int i = 0;
	//std::ofstream osobniki(m_config.outputFolderPath.string() + "\\details\\gen__" + std::to_string(m_generationNumber) + ".txt");
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
	//		                                                         m_algorithmConfig.m_svmConfig.m_height,
	//		                                                         m_algorithmConfig.m_svmConfig.m_width,
	//		                                                         *m_trainingSet, *m_validationSet);
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
	//		osobniki << "Gammas tr: ";
	//		for (auto g : individual.getDataset())
	//		{
	//			osobniki << g.gamma << ", ";
	//		}
	//		for (auto v : m_frozenSV_ids)
	//		{
	//			osobniki << v.gamma << ", ";
	//		}
	//		osobniki << std::endl;
	//		
	//		osobniki << "Gammas: ";
	//		for (auto j = 0; j < res->m_model->l; ++j)
	//			osobniki << res->m_model->param.gammas_after_training->at(j) << " ";
	//		osobniki << std::endl;

	//		osobniki << "Alphas: ";
	//		for (auto j = 0; j < res->m_model->l; ++j)
	//		{
	//			osobniki << res->m_model->sv_coef[0][j] << " ";
	//		}
	//		osobniki << "\nTraining chromosome: ";
	//		for (auto j = 0; j < individual.getDataset().size(); ++j)
	//		{
	//			osobniki << individual.getDataset()[j].gamma << ":" << individual.getDataset()[j].classValue << "  ";
	//		}
	//		osobniki << "\n";
	//		osobniki << "Rho: " << res->m_model->rho[0] << "\n";
	//		osobniki << "thr: " << res->m_optimalProbabilityThreshold << "\n";
	//	}
	//}
	//osobniki.close();
}

void RbfLinearWorkflow::visualizeFrozenSet(geneticComponents::Population<RbfLinearWorkflow::SvmCustomKernelChromosome>& best_pop)
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

	auto image3 = visualization3.createDetailedVisualization(*best_pop.getBestOne().getClassifier(), 500, 500, *m_trainingSet, *m_trainingSet, *m_testSet);
	SvmWokrflowConfiguration config_copy3{"", "", "", m_config.outputFolderPath, "m_supportVectorFrozenPool", ""};
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, m_generationNumber);
	m_savePngElement.launch(image3, m_pngNameSource);
}

bool RbfLinearWorkflow::shrinkTrainingSet(geneticComponents::Population<RbfLinearWorkflow::SvmCustomKernelChromosome>& best_pop)
{
	dataset::Dataset<std::vector<float>, float> new_Training;
	std::unordered_set<uint64_t> new_training_set_ids;
	auto samples = m_trainingSet->getSamples();
	auto labels = m_trainingSet->getLabels();
	//auto classifier = m_population.getBestOne().getClassifier();
	//auto classifier = best_pop.getBestOne().getClassifier();  //m_population.getBestOne().getClassifier();

	//if (m_currentGamma == -1) //for linear only case
	{
		auto classifier = m_population.getBestOne().getClassifier();
		for (auto f = 0; f < samples.size(); ++f)
		{
			float predicted = -100;
			if(classifier->canClassifyWithOptimalThreshold())
			{
				predicted = static_cast<float>(classifier->classifyWithOptimalThreshold((samples[f])));
			}
			else
			{
				predicted = classifier->classify(samples[f]);
			}
			if (predicted == labels[f])
			{
				continue;
			}
			else
			{
				new_training_set_ids.emplace(f);
				new_Training.addSample(samples[f], labels[f]);
			}
		}
		m_trainingSet2 = new_Training;
	}
	//else
	//{
	//	for (auto& individual : m_population)
	//	{
	//		auto classifier = individual.getClassifier();
	//		for (auto f = 0; f < samples.size(); ++f)
	//		{
	//			if (classifier->classifyWithOptimalThreshold(samples[f]) == labels[f])
	//			//if (classifier->classify(samples[f]) == labels[f])
	//			{
	//				continue;
	//			}
	//			else
	//			{
	//				new_training_set_ids.emplace(f);
	//				new_Training.addSample(samples[f], labels[f]);
	//			}
	//		}
	//	}
	//}
	

	std::unordered_set<uint64_t> forbidden_set;
	for (auto j = 0; j < samples.size(); ++j)
	{
		if (new_training_set_ids.find(j) == new_training_set_ids.end())
		{
			forbidden_set.emplace(j);
		}
	}

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

	m_forbidden_set = forbidden_set;
	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setForbbidens(forbidden_set);
	reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setForbbidens(forbidden_set);
	m_crossoverCompensationElement.setForbbidens(forbidden_set);

	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setImbalancedOrOneClass(true);
	m_crossoverCompensationElement.setImbalancedOrOneClass(true);
	reinterpret_cast<const std::shared_ptr<svmComponents::CrossoverCustomGauss>&>(m_algorithmConfig.m_crossover)->setImbalancedOrOneClass(true);
	m_superIndividualsGenerationElement->setImbalancedOrOneClass(true);
	return false;
}


void RbfLinearWorkflow::runGeneticAlgorithm()
{
	auto previousIterationFitness = 0.0;
	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> popBackup;
	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> popBackupTestScore;

	//linear first
	{
		m_currentGamma = -1;
		//m_CValue = 0.01; //TODO make this work differently

		GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	                                                                       1,
	                                                                       1,
	                                                                       m_numberOfClassExamples,
	                                                                       1,
	                                                                       std::make_shared<svmComponents::SvmKernelTraining>(
		                                                                       m_algorithmConfig.m_svmConfig,
		                                                                       m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::
		                                                                       Auc),
	                                                                       std::make_shared<svmComponents::LinearKernel>(cv::ml::ParamGrid(0.001, 1050, 10),
	                                                                                                                  false)),
	                      m_loadingWorkflow);

		auto bestOne = gs.run();
		m_CValue = bestOne->getC();

		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setNumberOfClassExamples(m_numberOfClassExamples);
		m_adaptationElement.resetToInitial(m_numberOfClassExamples);

		m_superIndividualsGenerationElement->setC(m_CValue);
		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->setCandGammaSingle(
			m_CValue, m_currentGamma);
		reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_currentGamma);
		m_crossoverCompensationElement.setGamma(m_currentGamma);
		m_supportVectorPoolElement.setCurrentGamma(m_currentGamma);

		initMemetic();
		memeticAlgorithm();

		if (m_population.getBestOne().getFitness() == 1.0)
			return;


		
		
		auto bestVectors = m_population.getBestOne().getDataset();
		for (auto& g : bestVectors)
		{
			m_frozenSV_ids.insert(g);
		}
		m_frozenSV.clear();
		m_frozenSV.insert(std::end(m_frozenSV), std::begin(m_frozenSV_ids), std::end(m_frozenSV_ids));

		//visualize the best vectors - train the best one and plot the results
		geneticComponents::Population<SvmCustomKernelChromosome> best_pop;
		auto copy2 = m_frozenSV;
		svmComponents::SvmCustomKernelChromosome best_vec{ std::move(copy2), m_population.getBestOne().getC() };
		best_pop = { std::vector<SvmCustomKernelChromosome>{best_vec} };
		m_trainingSvmClassifierElement.launch(best_pop, *m_trainingSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			visualizeFrozenSet(best_pop);
		}

		//shrink training set
		if (m_shrinkTrainingSet)
		{
			shrinkTrainingSet(best_pop);
		}

		popBackup = m_population;
		auto copy = m_population;
		auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);
		popBackupTestScore = testPopulation;
	}

	switchMetric(); //switch metric from Accuracy to AUC after getting proper linear vectors

	//auto metric = std::make_shared<svmComponents::SvmAucMetric>();
	//GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	//                                                                       1,
	//                                                                       1,
	//                                                                       m_numberOfClassExamples,
	//                                                                       1,
	//                                                                       std::make_shared<svmComponents::SvmKernelTraining>(
	//	                                                                       m_algorithmConfig.m_svmConfig,
	//	                                                                       m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::
	//	                                                                       Auc),
	//                                                                       std::make_shared<svmComponents::RbfKernel>(cv::ml::ParamGrid(0.001, 1050, 10),
	//                                                                                                                  cv::ml::ParamGrid(0.001, 1050, 10),
	//                                                                                                                  false)),
	//                      m_loadingWorkflow);

	//svmComponents::GaSvmGenerationWithForbbidenSet generation(*m_trainingSet,
	//                                                          std::make_unique<random::MersenneTwister64Rng>(0),
	//                                                          m_numberOfClassExamples,
	//                                                          svmComponents::svmUtils::countLabels(static_cast<unsigned int>(2), *m_trainingSet)); //TODO multiclass support
	//generation.setForbbidens(m_forbidden_set);
	//generation.setImbalancedOrOneClass(true);
	////gs.switchMetric(metric); //important switch metric to AUC in here 
	//
	//auto bestOne = gs.runWithGeneration(generation);

	
	m_gammaRange = { 0.001, 0.01, 0.1, 1, 10, 100, 1000};
	std::cout << m_CValue;

	std::vector<double> gamma_fitness;
	for (auto i = 0u; i < m_gammaRange.size(); ++i)
	{
		m_currentGamma = m_gammaRange[i];

		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setNumberOfClassExamples(m_numberOfClassExamples);
		m_adaptationElement.resetToInitial(m_numberOfClassExamples);

		m_superIndividualsGenerationElement->setC(m_CValue);
		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->setCandGammaSingle(
			m_CValue, m_gammaRange[i]);
		reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
		m_crossoverCompensationElement.setGamma(m_gammaRange[i]);
		m_supportVectorPoolElement.setCurrentGamma(m_gammaRange[i]);


		initMemetic();
		gamma_fitness.emplace_back(m_population.getBestOne().getFitness());
	}

	/*for(auto i = 0u; i < m_gammaRange.size(); ++i)
	{
		std::cout << "Gamma: " << m_gammaRange[i] << "  Fitness: " << gamma_fitness[i] << "\n";
	}*/

	auto bestGamma = m_gammaRange[std::max_element(gamma_fitness.begin(), gamma_fitness.end()) - gamma_fitness.begin()];
	m_gammaRange = { bestGamma / 5,  bestGamma, bestGamma * 10, bestGamma * 50};
	
	//for given gamma range
	double initialBest = m_population.getBestOne().getFitness();
	for (auto i = 0u; i < m_gammaRange.size(); ++i)
	{
		m_currentGamma = m_gammaRange[i];

		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
				setNumberOfClassExamples(m_numberOfClassExamples);
		m_adaptationElement.resetToInitial(m_numberOfClassExamples);

		m_superIndividualsGenerationElement->setC(m_CValue);
		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->setCandGammaSingle(
			m_CValue, m_gammaRange[i]);
		reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
		m_crossoverCompensationElement.setGamma(m_gammaRange[i]);
		m_supportVectorPoolElement.setCurrentGamma(m_gammaRange[i]);


		initMemetic();


		if (m_population.getBestOne().getFitness() - initialBest < 0.0)
		{
			m_population = popBackup;
			logResults(m_population, popBackupTestScore);

			continue;
		}

		memeticAlgorithm();

		//switchMetric(); //switch metric from Accuracy to AUC after getting proper linear vectors
		
		if (m_population.getBestOne().getFitness() - previousIterationFitness < 0.0)
		{
			m_population = popBackup;
			logResults(m_population, popBackupTestScore);
			continue;
		}


		previousIterationFitness = m_population.getBestOne().getFitness();
		//sort of early stopping
		if (m_population.getBestOne().getFitness() == 1.0)
			break;

		//auto copy = m_svPool;
		/*auto copy = m_supportVectorPool2;
		svmComponents::SvmCustomKernelChromosome best_vec{ std::move(copy), m_CValue };
		geneticComponents::Population<SvmCustomKernelChromosome> best_pop2{ std::vector<SvmCustomKernelChromosome>{best_vec} };
		m_trainingSvmClassifierElement.launch(best_pop2, *m_trainingSet);

		svmComponents::SvmVisualization visualization3;
		auto image3 = visualization3.createDetailedVisualization(*best_pop2.getBestOne().getClassifier(), 500, 500, *m_trainingSet, *m_trainingSet);
		SvmWokrflowConfiguration config_copy3{ "","","",m_config.outputFolderPath, "m_supportVectorPool_individual!!!!!!!!","" };
		setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, m_generationNumber);
		m_savePngElement.launch(image3, m_pngNameSource);*/

		//auto bestVectors = getBestSupportVector2(best_pop2.getBestOne(), *m_validationSet, *m_trainingSet, 0.8); //score equal to 1.0
		//auto bestVectors = m_supportVectorPool2;

		//froze the best vectors 
		//auto bestVectors = getBestSupportVector2(m_population.getBestOne(), *m_validationSet, *m_trainingSet, 0.0);
		auto bestVectors = m_population.getBestOne().getDataset();

		for (auto& g : bestVectors)
		{
			m_frozenSV_ids.insert(g);
		}
		m_frozenSV.clear();
		m_frozenSV.insert(std::end(m_frozenSV), std::begin(m_frozenSV_ids), std::end(m_frozenSV_ids));

		//visualize the best vectors - train the best one and plot the results
		geneticComponents::Population<SvmCustomKernelChromosome> best_pop;
		auto copy2 = m_frozenSV;
		svmComponents::SvmCustomKernelChromosome best_vec{std::move(copy2), m_population.getBestOne().getC()};
		best_pop = {std::vector<SvmCustomKernelChromosome>{best_vec}};
		m_trainingSvmClassifierElement.launch(best_pop, *m_trainingSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			visualizeFrozenSet(best_pop);
		}

		//shrink training set
		if (m_shrinkTrainingSet)
		{
			if (shrinkTrainingSet(best_pop))
				break;
		}


		if (!m_population.empty())
		{
			initialBest = m_population.getBestOne().getFitness();
		}

		popBackup = m_population;
		auto copy = m_population;		
		auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);
		popBackupTestScore = testPopulation;
	}
}

void RbfLinearWorkflow::logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
                                   const geneticComponents::Population<SvmCustomKernelChromosome>& testPopulation)
{
	auto bestOneConfustionMatrix = population.getBestOne().getConfusionMatrix().value();
	auto validationDataset = *m_validationSet;
	auto featureNumber = validationDataset.getSamples()[0].size();
	auto bestOneIndex = m_population.getBestIndividualIndex();

	m_resultLogger.createLogEntry(population,
	                              testPopulation,
	                              m_timer,
	                              m_algorithmName,
	                              m_generationNumber,
	                              svmComponents::Accuracy(bestOneConfustionMatrix),
	                              featureNumber,
	                              m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size() + m_frozenSV.size(),
	                              bestOneConfustionMatrix,
	                              testPopulation[bestOneIndex].getConfusionMatrix().value());
	m_generationNumber++;
}

void RbfLinearWorkflow::initMemetic()
{
	try
	{
		m_svPool.clear();
		m_supportVectorPoolElement.clear();
		//m_numberOfClassExamples = m_initialNumberOfClassExamples;

		
		
		if (m_currentGamma != -1)
		{
			std::call_once(m_increasAfterKernelSwitch,
			               [&, this]()
			               {
				               m_adaptationElement.resetToInitial(2 * m_initialNumberOfClassExamples);
			               });
		}
		m_adaptationElement.setFrozenSetSize(static_cast<unsigned int>(m_frozenSV.size()));

		auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);

		auto tr = reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernel>&>(m_algorithmConfig.m_training);
		tr->trainPopulation(population, *m_trainingSet, m_frozenSV);

		m_population = m_valdiationElement->launch(population, *m_validationSet);

		for (auto& individual : m_population)
		{
			auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(individual.getClassifier().get());
			if (res2->isMaxIterReached() && m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				svmComponents::SvmVisualization visualization2;
				visualization2.setGene(individual);
				visualization2.setGammasValues(m_gammaRange);
				visualization2.setFrozenSet(m_frozenSV);
				auto image2 = visualization2.createDetailedVisualization(*individual.getClassifier(), 500, 500, *m_trainingSet, *m_trainingSet, *m_testSet);
				SvmWokrflowConfiguration config_copy2{"", "", "", m_config.outputFolderPath, "max_iter_Problem", ""};
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, m_generationNumber);
				m_savePngElement.launch(image2, m_pngNameSource);
			}
		}
		internalVisualization_Debugging(m_population);

		auto testPopulation = m_valdiationTestDataElement->launch(population, *m_testSet);

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
			                                                       *m_trainingSet, *m_validationSet, *m_testSet);

			m_savePngElement.launch(image, m_pngNameSource);
		}

		//logAllModels(testPopulation);
		logResults(m_population, testPopulation);
	}
	catch(const svmComponents::EmptySupportVectorPool& exception)
	{
		LOG_F(ERROR, "With gamma %f  Error: %s", m_currentGamma, exception.what());
		std::cout << exception.what();
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
		throw;
	}
}

void RbfLinearWorkflow::memeticAlgorithm()
{
	try
	{
		bool isStop = false;

		while (!isStop)
		{
			auto parents = m_parentSelectionElement.launch(m_population);
			auto newPopulation = m_crossoverElement.launch(parents);

			auto compensantionInfo = m_compensationGenerationElement.generate(parents, m_numberOfClassExamples);
			auto result = m_crossoverCompensationElement.compensate(newPopulation, compensantionInfo);

			m_educationElement->educatePopulation(result, m_svPool, parents, *m_trainingSet);
			auto populationEducated = result;

			populationEducated = m_mutationElement.launch(populationEducated);

			auto tr = reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernel>&>(m_algorithmConfig.m_training);

			tr->trainPopulation(populationEducated, *m_trainingSet, m_frozenSV);
			auto poptrained = populationEducated;
			auto afterValidtion = m_valdiationElement->launch(populationEducated, *m_validationSet);

			internalVisualization_Debugging(populationEducated);

			m_supportVectorPoolElement.updateSupportVectorPool(poptrained, *m_trainingSet);
			m_svPool = m_supportVectorPoolElement.getSupportVectorPool();

			auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
			auto superIndividualsPopulation = m_superIndividualsGenerationElement->createPopulation(superIndividualsSize, m_svPool, m_numberOfClassExamples);

			tr->trainPopulation(superIndividualsPopulation, *m_trainingSet, m_frozenSV);
			m_validationSuperIndividualsElement->launch(superIndividualsPopulation, *m_validationSet);

			auto combinedPopulation = m_populationCombinationElement.launch(afterValidtion, superIndividualsPopulation);
			m_population = m_selectionElement.launch(m_population, combinedPopulation);

			m_adaptationElement.adapt(m_population);
			auto IsModeLocal = m_adaptationElement.getIsModeLocal();
			auto NumberOfClassExamples = m_adaptationElement.getNumberOfClassExamples();
			m_numberOfClassExamples = NumberOfClassExamples;

			auto copy = m_population;
			auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);

			isStop = m_stopConditionElement.launch(m_population);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);

				auto svm = m_population.getBestOne().getClassifier();
				svmComponents::SvmVisualization visualization;
				auto best = m_population.getBestOne();

				visualization.setGene(best);
				visualization.setGammasValues(m_gammaRange);
				visualization.setFrozenSet(m_frozenSV);

				auto image = visualization.createDetailedVisualization(*svm,
				                                                       m_algorithmConfig.m_svmConfig.m_height,
				                                                       m_algorithmConfig.m_svmConfig.m_width,
				                                                       *m_trainingSet, *m_validationSet, *m_testSet);

				m_savePngElement.launch(image, m_pngNameSource);
			}

			auto localGlobal = dynamic_cast<geneticComponents::LocalGlobalAdaptationSelection<SvmCustomKernelChromosome>*>(m_algorithmConfig
			                                                                                                               .m_parentSelection.get());
			if (localGlobal != nullptr)
			{
				localGlobal->setMode(IsModeLocal);
			}

			svmComponents::SvmAucprcMetric metric;
		
			
			//logAllModels(testPopulation);
			logResults(m_population, testPopulation);

			if (m_shrinkTrainingSet && !m_algorithmConfig.m_trainAlpha)
			{
				if (shrinkTrainingSet(m_population))
					continue;
			}

		/*	auto best_copy = m_population.getBestOne();
			auto wynik = metric.calculateMetric(best_copy, *m_validationSet);
			std::cout << "PRAUC: " << wynik.m_fitness << "\n";*/
		}
	}
	catch (const svmComponents::EmptySupportVectorPool& exception)
	{
		LOG_F(ERROR, "With gamma %f  Error: %s", m_currentGamma, exception.what());
		std::cout << exception.what();
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
		throw;
	}
}

void RbfLinearWorkflow::switchMetric()
{
	m_estimationMethod = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc);
	m_valdiationTestDataElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod, true);
	m_valdiationElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod, false);
	m_validationSuperIndividualsElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod, false);
}
} // namespace genetic
