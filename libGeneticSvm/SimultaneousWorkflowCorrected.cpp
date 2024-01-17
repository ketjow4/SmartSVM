#include "SimultaneousWorkflowCorrected.h"
#include "libSvmComponents/SvmSimultaneousChromosome.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "AllModelsLogger.h"
#include "LibGeneticComponents/CrossoverSelectionFactory.h"
#include "libPlatform/StringUtils.h"

namespace genetic
{
SimultaneousWorkflowCorrected::SimultaneousWorkflowCorrected(const SvmWokrflowConfiguration& config,
                                                             SimultaneousWorkflowConfig algorithmConfig,
                                                             IDatasetLoader& workflow)
	: m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
	, m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
	, m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
	, m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
	, m_algorithmConfig(std::move(algorithmConfig))
	, m_svmTraining(*m_algorithmConfig.m_svmTraining)
	, m_validation(*algorithmConfig.m_svmConfig.m_estimationMethod, false)
	, m_validationTest(*algorithmConfig.m_svmConfig.m_estimationMethod, true)
	, m_stopConditionElement(*algorithmConfig.m_stopCondition)
	, m_trainingSet(workflow.getTraningSet())
	, m_validationSet(workflow.getValidationSet())
	, m_testSet(workflow.getTestSet())
	, m_selectionElement(*m_algorithmConfig.m_selectionElement)
	, m_generationNumber(0)
	, m_workflow(workflow)
	, m_config(config)
	, m_timer(std::make_shared<Timer>())
{
}

void SimultaneousWorkflowCorrected::logAllModels(AllModelsLogger& logger)
{
	auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
	auto bestOneIndex = m_pop.getBestIndividualIndex();
	auto testbestOneConfustionMatrix = m_popTestSet[bestOneIndex].getConfusionMatrix().value();
	logger.log(m_pop,
		m_popTestSet,
		*m_timer,
		m_algorithmName,
		m_generationNumber++,
		Accuracy(bestOneConfustionMatrix),
		m_pop.getBestOne().featureSetSize(),
		bestOneConfustionMatrix,
		testbestOneConfustionMatrix);
}

std::shared_ptr<phd::svm::ISvm> SimultaneousWorkflowCorrected::run()
{
	static unsigned int numberOfRun = 1;
	auto outputPaht = m_config.outputFolderPath.string();
	AllModelsLogger logger{numberOfRun, outputPaht, m_workflow};

	std::vector<svmComponents::BaseSvmChromosome> classifier;
	std::vector<std::string> logentries;

	for (int i = 0; i < 5; i++)
	{
		init();
		m_svmTraining.launch(m_pop, m_trainingSet);
		evaluate();
		logAllModels(logger);

		while (!isFinished())
		{
			performEvolution();
			evaluate();

			logAllModels(logger);
		}

		classifier.push_back(m_pop.getBestOne()); //return best in here
		logentries.push_back(*m_resultLogger.getLogEntries().crbegin());
		m_resultLogger.logToFile(m_resultFilePath);
		m_resultLogger.clearLog();
	}

	if (numberOfRun == 1)
	{
		//adding time twice so we need to decrease it in here on the first run
		std::ifstream timeOfInitPython(m_config.outputFolderPath.string() + "\\timeOfEnsembleFeatures.txt");
		double time = 0.0;
		timeOfInitPython >> time;
		std::cout << time << "\n";
		m_timer->decreaseTime(time * 1000); //converting to miliseconds
	}

	if (numberOfRun != 1)
	{
		std::cout << "SESVM Adding time of feature selection\n";
		std::ifstream timeOfInitPython(m_config.outputFolderPath.string() + "\\timeOfEnsembleFeatures.txt");
		double time = 0.0;
		timeOfInitPython >> time;
		std::cout << time << "\n";
		m_timer->addTime(time * 1000); //converting to miliseconds
	}

	auto bestOne = std::max_element(classifier.begin(), classifier.end(),
	                                [](svmComponents::BaseSvmChromosome left, svmComponents::BaseSvmChromosome right)
	                                {
		                                return left.getFitness() < right.getFitness();
	                                });
	auto it = std::find_if(classifier.begin(), classifier.end(),
	                       [&bestOne](svmComponents::BaseSvmChromosome element)
	                       {
		                       return element.getFitness() == bestOne->getFitness();
	                       });
	auto pos = std::distance(classifier.begin(), it);
	std::vector<std::string> a;

	auto finalEntry = ::platform::stringUtils::splitString(logentries[pos], '\t');

	auto time = ::platform::stringUtils::splitString(*logentries.rbegin(), '\t')[3];

	//finalEntry[3] = time;
	finalEntry[3] = std::to_string(m_timer->getTimeMiliseconds().count());

	std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
	                                [](std::string& ss, std::string& s)
	                                {
		                                return ss.empty() ? s : ss + "\t" + s;
	                                });
	//s.append("\n");

	a.push_back(s);

	logger.save(outputPaht + "\\" + std::to_string(numberOfRun) + "\\populationTextLog.txt");
	//m_resultLogger.logToFile(m_resultFilePath);

	m_resultLogger.setEntries(a);
	m_resultLogger.logToFile(m_resultFilePath);

	numberOfRun++;

	return bestOne->getClassifier(); //m_pop.getBestOne().getClassifier(); //return best in here
}

void SimultaneousWorkflowCorrected::init()
{
	auto popSize = m_algorithmConfig.m_populationSize;
	auto features = m_featureSetOptimization->initNoEvaluate(popSize);
	auto kernels = m_kernelOptimization->initNoEvaluate(popSize);
	auto traningSets = m_trainingSetOptimization->initNoEvaluate(popSize);

	std::vector<svmComponents::SvmSimultaneousChromosome> vec;
	vec.reserve(popSize);
	for (auto i = 0u; i < features.size(); ++i)
	{
		svmComponents::SvmSimultaneousChromosome c{kernels[i], traningSets[i], features[i]};
		vec.emplace_back(c);
	}
	geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{vec};
	m_pop = pop;
}

geneticComponents::Population<svmComponents::SvmKernelChromosome> SimultaneousWorkflowCorrected::getKernelPopulation()
{
	std::vector<svmComponents::SvmKernelChromosome> vec;
	vec.reserve(m_pop.size());
	for (auto i = 0u; i < m_pop.size(); ++i)
	{
		svmComponents::SvmKernelChromosome c{m_pop[i].getKernelType(), m_pop[i].getKernelParameters(), false};
		vec.emplace_back(c);
	}
	geneticComponents::Population<svmComponents::SvmKernelChromosome> pop{vec};
	return pop;
}

geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> SimultaneousWorkflowCorrected::getFeaturesPopulation()
{
	std::vector<svmComponents::SvmFeatureSetMemeticChromosome> vec;
	vec.reserve(m_pop.size());
	for (auto i = 0u; i < m_pop.size(); ++i)
	{
		svmComponents::SvmFeatureSetMemeticChromosome c{m_pop[i].getFeaturesChromosome()};
		vec.emplace_back(c);
	}
	geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> pop{vec};
	return pop;
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> SimultaneousWorkflowCorrected::getTrainingSetPopulation()
{
	std::vector<svmComponents::SvmTrainingSetChromosome> vec;
	vec.reserve(m_pop.size());
	for (auto i = 0u; i < m_pop.size(); ++i)
	{
		svmComponents::SvmTrainingSetChromosome c{m_pop[i].getTraining()};
		vec.emplace_back(c);
	}
	geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> pop{vec};
	return pop;
}

void SimultaneousWorkflowCorrected::setKernelPopulation(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
                                                        geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
	int i = 0;
	for (const auto& individual : population)
	{
		newPop[i].setKernel(individual);
		++i;
	}
}

void SimultaneousWorkflowCorrected::setTrainingPopulation(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
                                                          geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
	int i = 0;
	for (const auto& individual : population)
	{
		newPop[i].setTraining(individual);
		++i;
	}
}

void SimultaneousWorkflowCorrected::setFeaturesPopulation(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
                                                          geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
	int i = 0;
	for (const auto& individual : population)
	{
		newPop[i].setFeatures(individual);
		++i;
	}
}

void SimultaneousWorkflowCorrected::fixKernelPop(const geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& currentPop,
                                                 geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
	auto bestKernel = currentPop.getBestOne().getKernel();

	for (auto& individual : newPop)
	{
		if (individual.getKernelType() == phd::svm::KernelTypes::Custom)
		{
			individual.setKernel(bestKernel);
		}
	}
}


template<typename chromosome>
std::vector<Parents<chromosome>> selectFromIndexes(const std::vector<Indexes>& indexes, const Population<chromosome>& pop)
{
	std::vector<Parents<chromosome>> parents;
	
	for(auto& idx : indexes)
	{
		parents.emplace_back(pop[idx.first], pop[idx.second]);
	}

	return parents;
}


void SimultaneousWorkflowCorrected::performEvolution()
{
	auto kernels = getKernelPopulation();
	auto features = getFeaturesPopulation();
	auto trainingSets = getTrainingSetPopulation();


	//parents selection scheme creation from config
	//create parents selection scheme //TODO think about which algorithm scheme to use in here
	auto config_parent_selection = m_algorithmConfig.m_config.getNode("Svm." + m_algorithmConfig.m_config.getValue<std::string>("Svm.SSVM.KernelOptimization.Name"));
	auto m_parentSelection = CrossoverSelectionFactory::create<SvmSimultaneousChromosome>(config_parent_selection);

	//creation of proper vectors
	std::vector<Indexes> indexes; 
	for (auto i = 0; i < m_pop.size(); ++i)
	{
		auto idx = m_parentSelection->chooseIndexes(m_pop);
		indexes.emplace_back(idx);
	}

	//setting parents
	auto featureParents = selectFromIndexes(indexes, features);
	m_featureSetOptimization->setParents(featureParents);
	auto kernelParents = selectFromIndexes(indexes, kernels);
	m_kernelOptimization->setParents(kernelParents);
	auto trainingSetParents = selectFromIndexes(indexes, trainingSets);
	m_trainingSetOptimization->setParents(trainingSetParents);

	m_featureSetOptimization->performGeneticOperations(features);
	m_kernelOptimization->performGeneticOperations(kernels);
	m_trainingSetOptimization->performGeneticOperations(trainingSets);

	std::vector<svmComponents::SvmSimultaneousChromosome> vec;
	vec.reserve(m_algorithmConfig.m_populationSize);
	for (auto i = 0u; i < features.size(); ++i)
	{
		svmComponents::SvmSimultaneousChromosome c{};
		vec.emplace_back(c);
	}
	geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> newPopulation{vec};

	setFeaturesPopulation(features, newPopulation);
	setKernelPopulation(kernels, newPopulation);
	setTrainingPopulation(trainingSets, newPopulation);

	fixKernelPop(m_pop, newPopulation);

	m_svmTraining.launch(newPopulation, m_trainingSet);

	m_popTestSet = newPopulation; //copy in here!!!
	m_validation.launch(newPopulation, m_validationSet);

	auto m_pop2 = m_selectionElement.launch(m_pop, newPopulation);
	//m_algorithmConfig.m_selectionElement->selectNextGeneration()
	//select 
	m_pop = m_pop2;
}

void SimultaneousWorkflowCorrected::evaluate()
{
	m_popTestSet = m_pop; //copy in here!!!
	m_validation.launch(m_pop, m_validationSet);
	m_validationTest.launch(m_popTestSet, m_testSet);

	log();
}

bool SimultaneousWorkflowCorrected::isFinished()
{
	return m_stopConditionElement.launch(m_pop);
}

void SimultaneousWorkflowCorrected::log()
{
	//TODO add header to logs and modify python analysis to use this header
	auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
	auto bestOneIndex = m_pop.getBestIndividualIndex();
	auto testbestOneConfustionMatrix = m_popTestSet[bestOneIndex].getConfusionMatrix().value();


	//auto featureNumber = m_validationSet.getSamples()[0].size();

	m_resultLogger.createLogEntry(m_pop,
	                              m_popTestSet,
	                              *m_timer,
	                              m_algorithmName,
	                              m_generationNumber,
	                              Accuracy(bestOneConfustionMatrix),
	                              m_pop.getBestOne().featureSetSize(),
	                              m_pop.getBestOne().trainingSetSize() * 2, //TODO fix for multiclass
	                              bestOneConfustionMatrix,
	                              testbestOneConfustionMatrix);
}
} // namespace genetic
