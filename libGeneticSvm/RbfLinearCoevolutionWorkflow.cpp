#include "RbfLinearCoevolutionWorkflow.h"
#include "GridSearchWorkflow.h"
#include "libSvmComponents/RbfKernel.h"
#include "LibGeneticComponents/LocalGlobalAdaptationSelection.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "libSvmComponents/CustomKernelTraining.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "libPlatform/loguru.hpp"
#include "libSvmComponents/GaSvmGeneration.h"
#include "libSvmComponents/SvmAucMetric.h"
#include "libSvmComponents/SvmAucprcMetric.h"
#include "LibGeneticComponents/TruncationSelection.h"
#include "libSvmComponents/SvmSubsetValidation.h"
#include "libSvmComponents/SvmValidationFactory.h"

#include "libPlatform/Subprocess.h"
#include "TestApp/PythonPath.h"

namespace genetic
{
RbfLinearCoevolutionWorkflow::RbfLinearCoevolutionWorkflow(const SvmWokrflowConfiguration& config,
                                                           svmComponents::RbfLinearConfig algorithmConfig,
                                                           IDatasetLoader& workflow,
                                                           platform::Subtree subtreeConfig)
	: m_config(config)
	, m_algorithmConfig(std::move(algorithmConfig))
	, m_workflow(workflow)
	, m_subtreeConfig(subtreeConfig)
{
}

void runFeatureSelectionForTimeMeasure(std::filesystem::path treningSetPath)
{

	//filesystem::FileSystem fs;

	auto pythonScriptPath = std::filesystem::path("featureSelection.py");
	if (!std::filesystem::exists(pythonScriptPath))
	{
		throw platform::FileNotFoundException(pythonScriptPath.string());
	}

	const auto command = std::string(PYTHON_PATH + " featureSelection.py -t "
		+ treningSetPath.string());

	//std::cout << "Starting python script\n";

	//const auto [output, ret] = platform::subprocess::launchWithPipe(command);
	
}

std::shared_ptr<phd::svm::ISvm> RbfLinearCoevolutionWorkflow::run()
{
	//runFeatureSelectionForTimeMeasure("");
	
	auto rbf_linear = std::make_shared<CoevolutionHelper>(m_config, svmComponents::RbfLinearConfig(m_subtreeConfig, m_workflow.getTraningSet()), m_workflow, "RBF_LINEAR_C001", m_subtreeConfig);
	auto rbf_linear_C01 = std::make_shared<CoevolutionHelper>(m_config, svmComponents::RbfLinearConfig(m_subtreeConfig, m_workflow.getTraningSet()), m_workflow, "RBF_LINEAR_C01", m_subtreeConfig);
	auto rbf_linear_C1 = std::make_shared<CoevolutionHelper>(m_config, svmComponents::RbfLinearConfig(m_subtreeConfig, m_workflow.getTraningSet()), m_workflow, "RBF_LINEAR_C1", m_subtreeConfig);
	auto rbf_linear_C10 = std::make_shared<CoevolutionHelper>(m_config, svmComponents::RbfLinearConfig(m_subtreeConfig, m_workflow.getTraningSet()), m_workflow, "RBF_LINEAR_C10", m_subtreeConfig);
	auto rbf_linear_C100 = std::make_shared<CoevolutionHelper>(m_config, svmComponents::RbfLinearConfig(m_subtreeConfig, m_workflow.getTraningSet()), m_workflow, "RBF_LINEAR_C100", m_subtreeConfig);

	m_subtreeConfig.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", 8);
	m_subtreeConfig.putValue<std::string>("Svm.Metric", "AUC");
	auto rbf_multiGamma = std::make_shared<CoevolutionHelper>( m_config, svmComponents::RbfLinearConfig(m_subtreeConfig, m_workflow.getTraningSet()), m_workflow, "MULTI_GAMMA", m_subtreeConfig);

	
	m_population.emplace_back(rbf_linear);
	m_population.emplace_back(std::move(rbf_linear_C01));
	m_population.emplace_back(std::move(rbf_linear_C1));
	m_population.emplace_back(std::move(rbf_linear_C10));
	m_population.emplace_back(std::move(rbf_linear_C100));
	std::vector<double> c_values = { 0.01, 0.1, 1, 10, 100 };

	int xf = 0;
	for(auto& pop : m_population)
	{
		pop->initializeGeneticAlgorithm();
		pop->setGamma(-1);
		pop->setC(c_values[xf]);
		pop->initForGamma();
		pop->initMemetic();

		while (!pop->memeticAlgorithmSingleIteration())
			;

		pop->addToFrozenSet();
		pop->shrinkTrainingSetComplete();
		pop->switchMetric(); //switch metric from Accuracy to AUC after getting proper linear vectors
		pop->getGammaRangeRbfLinear();
		xf++;
		pop->backupForNoImprovement();
	}
	
	rbf_multiGamma->initializeGeneticAlgorithm();
	rbf_multiGamma->getGammasFromGridSearch();

	m_population.emplace_back(std::move(rbf_multiGamma));
	
	std::vector<bool> runIndicator = std::vector<bool>( m_population.size(), true );

	
	//while( !rbf_linear->gammasHasEnded() && !rbf_multiGamma->gammasHasEnded())
	while(true)
	{
		if (std::all_of(m_population.begin(), m_population.end(), [](std::shared_ptr<CoevolutionHelper> a) {return a->gammasHasEnded(); }))
			break;
	
		//for given gamma range
		int j = 0;
		for (auto& pop : m_population)
		{
			if(!pop->gammasHasEnded())
			{
				auto currentGamma = pop->getCurrentGamma();
				pop->setGamma(currentGamma);
				pop->initForGamma();
				pop->setInitialBest();
				pop->initMemetic();
			}
			else
			{
				runIndicator[j] = false;
				//std::cout << "Evolution ended for " << pop->m_algorithmName << "\n";
			}
			j++;
		}

		//select only 2 best subpopulations
		if(m_population.size() > 3)
		{
			std::sort(m_population.begin(), m_population.end(),
			          [](std::shared_ptr<CoevolutionHelper> a, std::shared_ptr<CoevolutionHelper> b)
			          {
				          return a->getAverageFitness() + a->getBest().getFitness() > b->getAverageFitness() + b->getBest().getFitness();
			          });
			m_population.erase(m_population.begin() + 3, m_population.end());

			runIndicator = std::vector<bool>(m_population.size(), true);
		}

		for (auto i = 0u; i < m_population.size(); i++)
		{
			if (!m_population[i]->improvementAfterInit())
				runIndicator[i] = false;
		}

		while (true)
		{

			for(auto i = 0u; i < m_population.size(); ++i)
			{
				if (runIndicator[i] && m_population[i]->memeticAlgorithmSingleIteration())
					runIndicator[i] = false;
			}

			//some coevolution in here

			if(std::none_of(runIndicator.begin(), runIndicator.end(), [](bool v) { return v; }))
			{
				std::sort(m_population.begin(), m_population.end(),
					[](std::shared_ptr<CoevolutionHelper> a, std::shared_ptr<CoevolutionHelper> b)
					{
						return a->getAverageFitness() > b->getAverageFitness();
					});

				for(auto i = 1; i < m_population.size(); ++i)
				{
					auto newSize = m_population[i]->getPopulationSize() - 2;

					if(newSize <= 2)
					{
						m_population.erase(m_population.begin() + i);
					}
					else
					{
						m_population[i]->setPopulationSize(newSize);
					}
				}
				break;
			}
		}

		std::vector<bool> runImproved = std::vector<bool>(m_population.size(), true);
		for (auto i = 0u; i < m_population.size(); i++)
		{
			if (!m_population[i]->improvementAfeterAlgorithm())
			{
				runImproved[i] = false;
			}
			else
			{
				m_population[i]->savePreviousIterFitness();
			}
		}

		if (std::any_of(m_population.begin(), m_population.end(), [](auto pop) {return pop->earlyStopAndPreviousBackup();}))
			break;
		
		for (auto i = 0u; i < m_population.size(); i++)
		{
			if(runImproved[i])
			{
				m_population[i]->addToFrozenSet();
				m_population[i]->shrinkTrainingSetComplete();
				m_population[i]->backupForNoImprovement();
			}
		}
		runIndicator = std::vector<bool>(m_population.size(), true);
	}

	std::sort(m_population.begin(), m_population.end(),
		[](std::shared_ptr<CoevolutionHelper> a, std::shared_ptr<CoevolutionHelper> b)
		{
			return a->getBest().getFitness() < b->getBest().getFitness();
		});

	if (m_config.verbosity != platform::Verbosity::None)
	{

		for (auto &pop : m_population)
		{
			pop->logToFile();
		}
	}
	//need to sort in descending for saving results
	std::sort(m_population.begin(), m_population.end(),
		[](std::shared_ptr<CoevolutionHelper> a, std::shared_ptr<CoevolutionHelper> b)
		{
			return a->getBest().getFitness() > b->getBest().getFitness();
		});
	
	return m_population[0]->getSolution();
}

CoevolutionHelper::CoevolutionHelper(const SvmWokrflowConfiguration& config,
                                     svmComponents::RbfLinearConfig algorithmConfig,
                                     IDatasetLoader& workflow,
									 std::string algorithmName,
									 platform::Subtree subtreeConfig)
	: m_algorithmConfig(std::move(algorithmConfig))
	, m_config(config)
	, m_loadingWorkflow(workflow)
	, m_trainingSet(nullptr)
	, m_validationSet(nullptr)
	, m_testSet(nullptr)
	, m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
	, m_valdiationElement(m_algorithmConfig.m_validationMethod)
	, m_valdiationTestDataElement(
		std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true))
	, m_validationSuperIndividualsElement(m_algorithmConfig.m_validationMethod)
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
	, m_parentSelectionElement(*m_algorithmConfig.m_parentSelection)
	, m_compensationGenerationElement(std::move(m_algorithmConfig.m_compensationGenerationElement))
	, m_numberOfClassExamples(m_algorithmConfig.m_numberOfClassExamples)
	, m_initialNumberOfClassExamples(m_algorithmConfig.m_numberOfClassExamples)
	, m_generationNumber(0)
	, m_subtreeConfig(subtreeConfig)
{
	m_algorithmName = algorithmName;
	m_hasEnded = false;
	m_shrinkTrainingSet = true;
	m_initialBest = 0.0;
	m_previousIterationFitness = 0.0;
	m_i = 0;
}

void CoevolutionHelper::logToFile()
{
	m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));
}

std::shared_ptr<phd::svm::ISvm> CoevolutionHelper::run()
{
	initializeGeneticAlgorithm();

	runGeneticAlgorithm();

	if(m_config.verbosity != platform::Verbosity::None)
	{
		logToFile();
	}
	return m_population.getBestOne().getClassifier();
}

void CoevolutionHelper::setC(double C)
{
	m_CValue = C;
}

void CoevolutionHelper::setGamma(double gamma)
{
	m_currentGamma = gamma;
}

void CoevolutionHelper::initializeGeneticAlgorithm()
{
	if (m_trainingSet == nullptr)
	{
		m_trainingSet = &m_loadingWorkflow.getTraningSet();
		m_validationSet = &m_loadingWorkflow.getValidationSet();
		m_testSet = &m_loadingWorkflow.getTestSet();
	}
}

void CoevolutionHelper::internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> /*pop*/)
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

void CoevolutionHelper::visualizeFrozenSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop)
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

bool CoevolutionHelper::shrinkTrainingSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop)
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
			if (classifier->canClassifyWithOptimalThreshold())
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

void CoevolutionHelper::addToFrozenSet()
{
	auto bestVectors = m_population.getBestOne().getDataset();
	for (auto& g : bestVectors)
	{
		m_frozenSV_ids.insert(g);
	}

	m_frozenSV.clear();
	m_frozenSV.insert(std::end(m_frozenSV), std::begin(m_frozenSV_ids), std::end(m_frozenSV_ids));
}

void CoevolutionHelper::shrinkTrainingSetComplete()
{
	geneticComponents::Population<SvmCustomKernelChromosome> best_pop;
	auto copy2 = m_frozenSV;
	svmComponents::SvmCustomKernelChromosome best_vec{std::move(copy2), m_population.getBestOne().getC()};
	best_pop = Population<svmComponents::SvmCustomKernelChromosome>(std::vector<SvmCustomKernelChromosome>{best_vec});
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
}

void CoevolutionHelper::getGammaRangeRbfLinear()
{
	m_gammaRange = {0.001, 0.01, 0.1, 1, 10, 100, 500};
	std::vector<double> gamma_fitness;
	for (auto i = 0u; i < m_gammaRange.size(); ++i)
	{
		m_currentGamma = m_gammaRange[i];

		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
				setNumberOfClassExamples(m_numberOfClassExamples);
		m_adaptationElement.resetToInitial(m_numberOfClassExamples);

		m_superIndividualsGenerationElement->setC(m_CValue);
		reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
				setCandGammaSingle(
					m_CValue, m_gammaRange[i]);
		reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange[i]);
		m_crossoverCompensationElement.setGamma(m_gammaRange[i]);
		m_supportVectorPoolElement.setCurrentGamma(m_gammaRange[i]);

		initMemetic();
		gamma_fitness.emplace_back(m_population.getBestOne().getFitness());
	}
	auto bestGamma = m_gammaRange[std::max_element(gamma_fitness.begin(), gamma_fitness.end()) - gamma_fitness.begin()];
	m_gammaRange = { bestGamma / 5, bestGamma, bestGamma * 10, bestGamma * 50};
}

bool CoevolutionHelper::improvementAfterInit()
{
	if (m_population.getBestOne().getFitness() - m_initialBest < 0.0)
	{
		m_population = m_popBackup;
		logResults(m_population, m_popBackupTestScore);

		return false;
	}
	return true;
}

bool CoevolutionHelper::improvementAfeterAlgorithm()
{
	if (m_population.getBestOne().getFitness() - m_previousIterationFitness < 0.0)
	{
		m_population = m_popBackup;
		logResults(m_population, m_popBackupTestScore);
		return false;
	}
	return true;
}

void CoevolutionHelper::setInitialBest()
{
	if (!m_population.empty())
	{
		m_initialBest = m_population.getBestOne().getFitness();
	}
	else
		m_initialBest = 0.0;
}

void CoevolutionHelper::backupForNoImprovement()
{
	m_popBackup = m_population;
	auto copy = m_population;
	auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);
	m_popBackupTestScore = testPopulation;
}

bool CoevolutionHelper::earlyStopAndPreviousBackup()
{
	if (m_population.getBestOne().getFitness() == 1.0)
		return true;
	return false;
}

void CoevolutionHelper::savePreviousIterFitness()
{
	m_previousIterationFitness = m_population.getBestOne().getFitness();
}

void CoevolutionHelper::getGammasFromGridSearch()
{
	GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
		1,
		1,
		m_numberOfClassExamples,
		1,
		std::make_shared<svmComponents::SvmKernelTraining>(
			m_algorithmConfig.m_svmConfig,
			m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::
			Auc),
		std::make_shared<svmComponents::RbfKernel>(svmComponents::ParamGrid(0.001, 1050, 10),
			svmComponents::ParamGrid(0.001, 1050, 10),
			false)),
		m_loadingWorkflow);

	auto bestOne = gs.run();

	try
	{
		m_gammaRange = { bestOne->getGamma() / 10, bestOne->getGamma(), bestOne->getGamma() * 10, bestOne->getGamma() * 100, bestOne->getGamma() * 1000 };
		m_CValue = bestOne->getC();
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
	}
}

void CoevolutionHelper::runGeneticAlgorithm()
{
	//linear first
	{
		m_currentGamma = -1;
		m_CValue = 0.01; //TODO make this work differently

		initForGamma();
		initMemetic();
		memeticAlgorithm();

		if (m_population.getBestOne().getFitness() == 1.0)
			return;

		addToFrozenSet();

		//visualize the best vectors - train the best one and plot the results
		shrinkTrainingSetComplete();
		
		backupForNoImprovement();
	}

	switchMetric(); //switch metric from Accuracy to AUC after getting proper linear vectors
	getGammaRangeRbfLinear();

	//for given gamma range
	for (auto i = 0u; i < m_gammaRange.size(); ++i)
	{
		m_currentGamma = m_gammaRange[i];
		initForGamma();
		initMemetic(true);

		if (!improvementAfterInit()) continue;

		memeticAlgorithm();

		if (! improvementAfeterAlgorithm()) continue;

		if (earlyStopAndPreviousBackup()) break;

		addToFrozenSet();

		//visualize the best vectors - train the best one and plot the results
		shrinkTrainingSetComplete();

		setInitialBest();
		backupForNoImprovement();
	}
}

void CoevolutionHelper::logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
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

void CoevolutionHelper::initForGamma()
{
	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setNumberOfClassExamples(m_numberOfClassExamples);
	m_adaptationElement.resetToInitial(m_numberOfClassExamples);

	m_superIndividualsGenerationElement->setC(m_CValue);
	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationRbfLinearSequential>&>(m_algorithmConfig.m_populationGeneration)->
			setCandGammaSingle(m_CValue, m_currentGamma);
	reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGaussSequential>&>(m_algorithmConfig.m_mutation)->setGamma(m_currentGamma);
	m_crossoverCompensationElement.setGamma(m_currentGamma);
	m_supportVectorPoolElement.setCurrentGamma(m_currentGamma);
}

void CoevolutionHelper::initMemetic(bool extendOnRbfToLinear)
{
	try
	{
		m_svPool.clear();
		m_supportVectorPoolElement.clear();

		if (m_currentGamma != -1 && extendOnRbfToLinear)
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

		/*for (auto& individual : m_population)
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
		}*/
		internalVisualization_Debugging(m_population);

		auto testPopulation = m_valdiationTestDataElement->launch(population, *m_testSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			setVisualizationFilenameAndFormatWithPrefix(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber, m_algorithmName);

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
	catch (const svmComponents::EmptySupportVectorPool& exception)
	{
		LOG_F(ERROR, "With gamma %f  Error: %s", m_currentGamma, exception.what());
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		throw;
	}
}

void CoevolutionHelper::memeticAlgorithm()
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
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		throw;
	}
}

bool CoevolutionHelper::memeticAlgorithmSingleIteration()
{
	try
	{
		bool isStop = false;

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
			if(superIndividualsSize == 0)
			{
				superIndividualsSize = 1;
			}
			auto superIndividualsPopulation = m_superIndividualsGenerationElement->createPopulation(superIndividualsSize, m_svPool, m_numberOfClassExamples);

			tr->trainPopulation(superIndividualsPopulation, *m_trainingSet, m_frozenSV);
			m_validationSuperIndividualsElement->launch(superIndividualsPopulation, *m_validationSet);

			auto combinedPopulation = m_populationCombinationElement.launch(afterValidtion, superIndividualsPopulation);
			m_population = m_selectionElement.launch(m_population, combinedPopulation);

			m_adaptationElement.setFrozenSetSize(static_cast<unsigned int>(m_frozenSV.size())); //TODO check this
			m_adaptationElement.adapt(m_population);
			auto IsModeLocal = m_adaptationElement.getIsModeLocal();
			auto NumberOfClassExamples = m_adaptationElement.getNumberOfClassExamples();
			m_numberOfClassExamples = NumberOfClassExamples;

			auto copy = m_population;
			auto testPopulation = m_valdiationTestDataElement->launch(copy, *m_testSet);

			isStop = m_stopConditionElement.launch(m_population);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				setVisualizationFilenameAndFormatWithPrefix(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber, m_algorithmName);

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
				shrinkTrainingSet(m_population);
			}
		}

		return isStop;
	}
	catch (const svmComponents::EmptySupportVectorPool& exception)
	{
		LOG_F(ERROR, "With gamma %f  Error: %s", m_currentGamma, exception.what());
		return true;
	}
	catch (const std::runtime_error& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());	
		throw;
	}
}

void CoevolutionHelper::switchMetric()
{
	m_estimationMethod = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc);
	m_valdiationTestDataElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod, true);

	

	if(m_valdiationElement->isUsingFullSet())
	{
		m_valdiationElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod, false);
		m_validationSuperIndividualsElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>>(*m_estimationMethod, false);	
	}
	else
	{
		//TODO fix ugly hack 
		auto config = m_subtreeConfig.getNode("Svm.RbfLinear");
		m_valdiationElement = svmComponents::SvmValidationFactory::create<SvmCustomKernelChromosome>(config, *m_estimationMethod);
		m_validationSuperIndividualsElement = svmComponents::SvmValidationFactory::create<SvmCustomKernelChromosome>(config, *m_estimationMethod);
	}	
}

int CoevolutionHelper::getPopulationSize()
{
	return static_cast<int>(m_algorithmConfig.m_populationSize);
}

void CoevolutionHelper::setPopulationSize(int popSize)
{
	m_algorithmConfig.m_populationSize = popSize;
	auto& selection = reinterpret_cast<geneticComponents::ConstantTruncationSelection<svmComponents::SvmCustomKernelChromosome>&>(m_selectionElement.getMethod());
	selection.setNewPopulationSize(popSize);
}
} // namespace genetic
