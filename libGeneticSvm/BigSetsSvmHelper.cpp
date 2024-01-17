#include "libPlatform/loguru.hpp"
#include "libPlatform/StringUtils.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "SvmLib/libSvmImplementation.h"
#include "BigSetsSvmHelper.h"

//DO NOT CHANGE ORDER OF INCLUDES
#include "LibGeneticComponents/CrossoverSelectionFactory.h"
#include "libSvmComponents/CustomKernelTraining.h"
#include "libSvmComponents/SvmAccuracyMetric.h"
#include "MemeticTrainingSetWorkflow.h"
#include "SvmEnsembleHelper.h"

namespace genetic
{
	BigSetsSvmHelper::BigSetsSvmHelper(const SvmWokrflowConfiguration& config,
		EnsembleTreeWorkflowConfig algorithmConfig,
		IDatasetLoader& workflow,
		bool addSvsToTraining,
		std::vector<DatasetVector>& SVs,
		IDatasetLoader& fullDatasetworkflow,
		std::vector<Gene> SvWithGamma,
		bool useDasvmKernel,
		bool debugLog,
		bool useFeatureSelection,
		platform::Subtree full_config,
		bool cascadeWideFeatureSelection,
		bool newDatasetFlow)
		: m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
		, m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
		, m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
		, m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
		, m_algorithmConfig(std::move(algorithmConfig))
		, m_svmTraining(*m_algorithmConfig.m_svmTraining)
		/*, m_validation(std::make_shared<svmStrategies::SvmValidationStrategy<SvmSimultaneousChromosome>>(*m_estimationMethod, false))
		, m_validationTest(std::make_shared<svmStrategies::SvmValidationStrategy<SvmSimultaneousChromosome>>(*m_estimationMethod, true))*/
		, m_stopConditionElement(*algorithmConfig.m_stopCondition)
		, m_trainingSet(workflow.getTraningSet())
		, m_validationSet(workflow.getValidationSet())
		, m_testSet(workflow.getTestSet())
		, m_selectionElement(*m_algorithmConfig.m_selectionElement)
		, m_generationNumber(0)
		, m_workflow(workflow)
		, m_config(config)
		, m_constKernel(phd::svm::KernelTypes::Linear, { 1 }, false) //TODO add to config
		//, m_constKernel(phd::svm::KernelTypes::Rbf, { 1, 1 } ,false)
		//, m_constKernel(phd::svm::KernelTypes::Rbf, { 1, 10 }, false)
		//, m_constKernel(phd::svm::KernelTypes::Rbf, { 1, 100 }, false)
		//, m_constKernel(phd::svm::KernelTypes::Rbf, { 1, 1000 }, false)
		, m_useConstKernel(algorithmConfig.m_constKernel)
		, m_addSvToTraining(addSvsToTraining)
		, m_svToAdd(SVs)
		, m_fullDatasetWorkflow(fullDatasetworkflow)
		, m_svFrozenPool(SvWithGamma)
		, m_useDasvmKernel(useDasvmKernel)
		, m_debugLog(debugLog)
		, m_useFeatureSelection(useFeatureSelection)
		, m_cascadeWideFeatureSelection(cascadeWideFeatureSelection)
		, m_newDatasetFlow(newDatasetFlow)
		, m_full_config(full_config)
	{
		static int nodeNumber = -1;
		nodeNumber++;

		m_nodeNumber = nodeNumber;

		//m_estimationMethod = m_algorithmConfig.m_svmConfig.m_estimationMethod;
		auto useBias = m_full_config.getValue<bool>("Svm.EnsembleTree.UseBias");
		auto useSingleClassPrediction = m_full_config.getValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds");
		m_estimationMethod = std::make_unique<SvmHyperplaneDistance>(false, useBias, useSingleClassPrediction);

		m_validation = std::make_shared<svmStrategies::SvmValidationStrategy<SvmSimultaneousChromosome>>(*m_estimationMethod, false);
		m_validationTest = std::make_shared<svmStrategies::SvmValidationStrategy<SvmSimultaneousChromosome>>(*m_estimationMethod, true);
	}

	void BigSetsSvmHelper::logAllModels(AllModelsLogger& logger)
	{
		auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
		logger.log(m_pop,
			m_popTestSet,
			m_timer,
			m_algorithmName,
			m_generationNumber++,
			Accuracy(bestOneConfustionMatrix),
			m_pop.getBestOne().featureSetSize(),
			bestOneConfustionMatrix);
	}

	void BigSetsSvmHelper::VisualizeWholePopulation(unsigned& /*numberOfRun*/)
	{
		//for (const auto& individual : m_pop)
		//{
		//       //auto individual = m_pop.getBestOne();
		//	//if (individual.getFitness() > 0.6)
		//	{
		//		filesystem::FileSystem fs;
		//		//auto detailsPath = m_config.outputFolderPath.string() + "\\details_" + std::to_string(numberOfRun) + "\\";
		//		auto detailsPath = m_config.outputFolderPath.string() + "\\BestOneHistory\\";
		//		fs.createDirectories(detailsPath);
		//		/*auto out = std::filesystem::path(detailsPath + timeUtils::getTimestamp() +
		//			"__gamma__" + std::to_string(individual.getClassifier()->getGamma()) +
		//			"__C__" + std::to_string(individual.getClassifier()->getC()) + "__iter__" +
		//			std::to_string(m_generationNumber) + "__fitness__" + std::to_string(individual.getFitness()) + ".png");*/

		//           auto out = std::filesystem::path(detailsPath + "gen__" +
		//               std::to_string(m_generationNumber) + "__" + timeUtils::getTimestamp() +
		//               "__gamma=" + std::to_string(individual.getClassifier()->getGamma()) +
		//               "__C=" + std::to_string(individual.getClassifier()->getC()) + "__fitness=" + std::to_string(individual.getFitness()) + ".png");
		//		
		//		auto svm = individual.getClassifier();
		//		svmComponents::SvmVisualization visualization;
		//		auto m_image = visualization.createDetailedVisualization(*svm,
		//		                                                         m_algorithmConfig.m_svmConfig.m_height,
		//		                                                         m_algorithmConfig.m_svmConfig.m_width,
		//		                                                         m_trainingSet, joinSets(m_fullDatasetWorkflow.getTraningSet(), m_fullDatasetWorkflow.getValidationSet()));

		//		m_savePngElement.launch(m_image, out);
		//	}
		//}
	}


	std::shared_ptr<phd::svm::ISvm> BigSetsSvmHelper::run()
	{
		unsigned int numberOfRun = 1;
		auto outputPaht = m_config.outputFolderPath.string();
		AllModelsLogger logger{ numberOfRun, outputPaht, m_workflow };

		std::vector<svmComponents::BaseSvmChromosome> classifier;
		std::vector<std::string> logentries;

		//for (int i = 0; i < 3; i++) TODO this was original before stability checking
		for (int i = 0; i < 1; i++)
		{
			try
			{

				init();
				train(m_pop);
				evaluate(m_pop);

				if (m_debugLog)
				{
					logAllModels(logger);
				}
				if (m_algorithmConfig.m_svmConfig.m_doVisualization) // only for debugging
				{
					VisualizeWholePopulation(numberOfRun);
				}

				while (!isFinished())
				{
					try
					{
						performEvolution();
						//evaluate();

						//update metric with proper distance information
						updateMetricDistance();

					}
					catch (const std::exception& e)
					{
						LOG_F(ERROR, "Error: %s", e.what());
						std::cout << e.what();
						throw;
					}

					if (m_debugLog)
					{
						logAllModels(logger);
					}

					if (m_algorithmConfig.m_svmConfig.m_doVisualization) // only for debugging
					{
						VisualizeWholePopulation(numberOfRun);
					}
				}

				classifier.push_back(m_pop.getBestOne()); //return best in here
				logentries.push_back(*m_resultLogger.getLogEntries().crbegin());
				m_resultLogger.logToFile(m_resultFilePath);
			}
			catch (const std::exception& e)
			{
				LOG_F(ERROR, "Error: %s", e.what());
				std::cout << e.what();
				throw;
			}
		}


		std::cout << "Best fitness:\n";
		auto i = 0;
		for (auto& c : classifier)
		{
			std::cout << "repeat: " << i++ << "  fit. " << c.getFitness() << "\n";
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

		finalEntry[3] = time;

		std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
			[](std::string& ss, std::string& s)
			{
				return ss.empty() ? s : ss + "\t" + s;
			});

		a.push_back(s);

		if (m_debugLog)
		{
			logger.save(outputPaht + "\\" + std::to_string(numberOfRun) + "\\populationTextLog.txt");
		}
		//m_resultLogger.logToFile(m_resultFilePath);

		m_resultLogger.setEntries(a);
		m_resultLogger.logToFile(m_resultFilePath);

		numberOfRun++;


		/*dataset::Dataset<std::vector<float>, float> dataset;
		if (m_newDatasetFlow)
		{
			dataset = joinSets(m_trainingSet, m_validationSet);
		}
		else
		{
			dataset = joinSets(m_trainingSet, m_fullDatasetWorkflow.getValidationSet());
		}
		auto setThresholds = dynamic_cast<SvmHyperplaneDistance*>(m_estimationMethod.get());
		setThresholds->calculateThresholds(*bestOne, dataset);
		auto fitness = setThresholds->calculateMetric(*bestOne, dataset, true);

		std::cout << fitness.m_fitness;

		svmComponents::SvmAccuracyMetric metric;
		auto result = metric.calculateMetric(*bestOne, dataset, true);
		std::cout << "Acc: " << result.m_fitness << " MCC: " << result.m_confusionMatrix->MCC() << "\n";*/


		auto fitnessMock = SvmSimultaneousChromosome();
		fitnessMock.updateFitness(bestOne->getFitness());
		m_pop = Population<svmComponents::SvmSimultaneousChromosome>{ std::vector<SvmSimultaneousChromosome>{fitnessMock} };

		return bestOne->getClassifier();
	}

	svmComponents::SvmTrainingSetChromosome BigSetsSvmHelper::getBestOne() const
	{
		return m_pop.getBestOne().getTraining();
	}

	void BigSetsSvmHelper::addVectorsToPopulation(std::vector<DatasetVector>& SVs)
	{
		auto trainingPop = getTrainingSetPopulation();

		for (auto& individual : trainingPop)
		{
			auto currentDataset = individual.getDataset();
			for (auto& sv : SVs)
			{
				currentDataset.emplace_back(sv);
			}
			individual.updateDataset(currentDataset);
		}
		setTrainingPopulation(trainingPop, m_pop);
	}

	void BigSetsSvmHelper::init()
	{
		try
		{
			auto popSize = m_algorithmConfig.m_populationSize;

			Population<SvmFeatureSetMemeticChromosome> features;
			if (m_useFeatureSelection)
			{
				features = m_featureSetOptimization->initNoEvaluate(popSize);
			}
			else
			{
				auto featureCount = m_trainingSet.getSample(0).size();
				std::vector<Feature> f;
				for (auto i = 0u; i < featureCount; ++i)
				{
					f.emplace_back(i);
				}

				std::vector<SvmFeatureSetMemeticChromosome> popTemp;
				for (auto i = 0u; i < popSize; ++i)
				{
					std::vector<Feature> temp;
					std::copy(f.begin(), f.end(), std::back_inserter(temp));
					popTemp.emplace_back(std::move(temp));
				}
				features = Population<SvmFeatureSetMemeticChromosome>(popTemp);
			}

			//auto kernels = m_kernelOptimization->initNoEvaluate(popSize);
			//auto traningSets = m_trainingSetOptimization->initNoEvaluate(popSize);
			//

			m_kernelOptimization->setupTrainingSet(m_trainingSet);
			auto kernels = m_kernelOptimization->initNoEvaluate(popSize, 0);
			auto traningSets = m_trainingSetOptimization->initNoEvaluate(popSize, 0);

			std::vector<svmComponents::SvmSimultaneousChromosome> vec;
			vec.reserve(popSize);
			svmComponents::SvmKernelChromosome constant = m_constKernel;

			for (auto i = 0u; i < traningSets.size(); ++i)
			{
				if (m_useConstKernel)
				{
					svmComponents::SvmSimultaneousChromosome c{ constant, traningSets[i], features[i] };
					vec.emplace_back(c);
				}
				else
				{
					svmComponents::SvmSimultaneousChromosome c{ kernels[i], traningSets[i], features[i] };
					vec.emplace_back(c);
				}
			}
			geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ vec };
			m_pop = pop;

			if (!m_useDasvmKernel)
			{
				if (m_addSvToTraining)
				{
					addVectorsToPopulation(m_svToAdd);
				}
			}
		}
		catch (const std::exception& e)
		{
			LOG_F(ERROR, "Error: %s", e.what());
			throw;
		}
	}

	geneticComponents::Population<svmComponents::SvmKernelChromosome> BigSetsSvmHelper::getKernelPopulation()
	{
		std::vector<svmComponents::SvmKernelChromosome> vec;
		vec.reserve(m_pop.size());
		for (auto i = 0u; i < m_pop.size(); ++i)
		{
			svmComponents::SvmKernelChromosome c{ m_pop[i].getKernelType(), m_pop[i].getKernelParameters(), false };
			vec.emplace_back(c);
		}
		geneticComponents::Population<svmComponents::SvmKernelChromosome> pop{ vec };
		return pop;
	}

	geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> BigSetsSvmHelper::getFeaturesPopulation()
	{
		if (m_useFeatureSelection)
		{

			std::vector<svmComponents::SvmFeatureSetMemeticChromosome> vec;
			vec.reserve(m_pop.size());
			for (auto i = 0u; i < m_pop.size(); ++i)
			{
				svmComponents::SvmFeatureSetMemeticChromosome c{ m_pop[i].getFeaturesChromosome() };
				vec.emplace_back(c);
			}
			geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> pop{ vec };
			return pop;
		}
		else
		{
			auto featureCount = m_trainingSet.getSample(0).size();
			std::vector<svmComponents::Feature> f;
			for (auto i = 0u; i < featureCount; ++i)
			{
				f.emplace_back(i);
			}
			svmComponents::SvmFeatureSetMemeticChromosome feature(std::move(f));

			std::vector<svmComponents::SvmFeatureSetMemeticChromosome> vec;
			vec.reserve(m_pop.size());
			for (auto i = 0u; i < m_pop.size() + 2; ++i) //TODO ugly fix here
			{
				svmComponents::SvmFeatureSetMemeticChromosome c{ feature };
				vec.emplace_back(c);
			}
			geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> pop{ vec };
			return pop;
		}
	}

	geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> BigSetsSvmHelper::getTrainingSetPopulation()
	{
		std::vector<svmComponents::SvmTrainingSetChromosome> vec;
		vec.reserve(m_pop.size());
		for (auto i = 0u; i < m_pop.size(); ++i)
		{
			svmComponents::SvmTrainingSetChromosome c{ m_pop[i].getTraining() };
			vec.emplace_back(c);
		}
		geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> pop{ vec };
		return pop;
	}

	void BigSetsSvmHelper::setKernelPopulation(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
	{
		int i = 0;
		for (const auto& individual : population)
		{
			newPop[i].setKernel(individual);

			if (m_useConstKernel)
			{
				svmComponents::SvmKernelChromosome constant = m_constKernel;
				newPop[i].setKernel(constant);
			}
			++i;
		}
	}

	void BigSetsSvmHelper::setTrainingPopulation(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
	{
		int i = 0;
		for (const auto& individual : population)
		{
			newPop[i].setTraining(individual);
			++i;
		}
	}

	void BigSetsSvmHelper::setFeaturesPopulation(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
	{
		int i = 0;
		for (const auto& individual : population)
		{
			newPop[i].setFeatures(individual);
			++i;
		}
	}

	void BigSetsSvmHelper::updateMetricDistance()
	{
		auto& svm = m_trainingSetOptimization->getClassifierWithBestDistances();
		if (&svm != nullptr && m_algorithmConfig.m_metricMode != defaultOption)
		{
			auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(&svm);

			auto useBias = m_full_config.getValue<bool>("Svm.EnsembleTree.UseBias");
			auto useSingleClassPrediction = m_full_config.getValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds");
			m_estimationMethod = std::make_shared<SvmHyperplaneDistance>(false, res->getPositiveNormalizedCertainty(), res->getNegativeNormalizedCertainty(),
				m_algorithmConfig.m_metricMode, useBias, useSingleClassPrediction);

			m_validation = std::make_shared<svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome>>(*m_estimationMethod, false);
			m_validationTest = std::make_shared<svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome>>(*m_estimationMethod, true);
		}
	}

	void BigSetsSvmHelper::fixKernelPop2(const geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& currentPop,
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
	{
		svmComponents::SvmKernelChromosome bestKernel;
		if (m_useConstKernel)
		{
			bestKernel = m_constKernel;
		}
		else
		{
			bestKernel = currentPop.getBestOne().getKernel();
		}

		for (auto& individual : newPop)
		{
			if (individual.getKernelType() == phd::svm::KernelTypes::Custom)
			{
				individual.setKernel(bestKernel);
			}
		}
	}

	template<typename chromosome>
	std::vector<Parents<chromosome>> selectFromIndexes2(const std::vector<Indexes>& indexes, const Population<chromosome>& pop)
	{
		std::vector<Parents<chromosome>> parents;

		for (auto& idx : indexes)
		{
			parents.emplace_back(pop[idx.first], pop[idx.second]);
		}

		return parents;
	}

	void BigSetsSvmHelper::performEvolution()
	{
		try
		{
			auto kernels = getKernelPopulation();
			auto features = getFeaturesPopulation();
			auto trainingSets = getTrainingSetPopulation();

			auto memetic = dynamic_cast<MemeticTraningSetWorkflow*>(m_trainingSetOptimization.get());
			memetic->getAdaptationElement().setFrozenSetSize(static_cast<unsigned int>(m_svFrozenPool.size()));

			auto config_parent_selection = m_algorithmConfig.m_config.getNode(
				"Svm." + m_algorithmConfig.m_config.getValue<std::string>(
					"Svm.EnsembleTree.KernelOptimization.Name"));
			auto m_parentSelection = CrossoverSelectionFactory::create<SvmSimultaneousChromosome>(config_parent_selection);

			//creation of proper vectors
			std::vector<Indexes> indexes;
			for (auto i = 0; i < m_pop.size(); ++i)
			{
				auto idx = m_parentSelection->chooseIndexes(m_pop);
				indexes.emplace_back(idx);
			}

			//setting parents
			auto featureParents = selectFromIndexes2(indexes, features);
			m_featureSetOptimization->setParents(featureParents);
			auto kernelParents = selectFromIndexes2(indexes, kernels);
			m_kernelOptimization->setParents(kernelParents);
			auto trainingSetParents = selectFromIndexes2(indexes, trainingSets);
			m_trainingSetOptimization->setParents(trainingSetParents);

			if (m_useFeatureSelection)
			{
				m_featureSetOptimization->performGeneticOperations(features);
			}
			m_kernelOptimization->performGeneticOperations(kernels);
			m_trainingSetOptimization->performGeneticOperations(trainingSets);

			std::vector<svmComponents::SvmSimultaneousChromosome> vec;
			vec.reserve(m_algorithmConfig.m_populationSize);
			for (auto i = 0u; i < trainingSets.size(); ++i)
			{
				svmComponents::SvmSimultaneousChromosome c{};
				vec.emplace_back(c);
			}
			geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> newPopulation{ vec };

			setFeaturesPopulation(features, newPopulation);
			setKernelPopulation(kernels, newPopulation);
			setTrainingPopulation(trainingSets, newPopulation);

			fixKernelPop2(m_pop, newPopulation);

			train(newPopulation);

			m_popTestSet = newPopulation; //copy in here!!!
			evaluate(newPopulation);

			auto m_pop2 = m_selectionElement.launch(m_pop, newPopulation);
			m_pop = m_pop2;
		}
		catch (const std::exception& e)
		{
			LOG_F(ERROR, "Error: %s", e.what());
			throw;
		}
	}




	void BigSetsSvmHelper::train(Population<SvmSimultaneousChromosome>& pop)
	{
		if (m_useDasvmKernel && m_full_config.getValue<bool>("Svm.EnsembleTree.NewDatasetSampling"))
		{
			//svmComponents::SvmTrainingCustomKernel dasvmTraining(m_algorithmConfig.m_svmConfig, false, "RBF_MAX", true);

			//std::vector<SvmCustomKernelChromosome> converted;

			//for (auto individual : pop)
			//{
			//	std::vector<Gene> genes;
			//	auto gamma = individual.getKernelParameters()[1]; //gamma is at index 1
			//	auto trainingSet = individual.getTraining().getDataset();
			//	for (auto tr : trainingSet)
			//	{
			//		genes.emplace_back(tr.id, tr.classValue, gamma); //convert training set
			//	}

			//	converted.emplace_back(std::move(genes), individual.getKernelParameters()[0], individual.getFeatures()); //C is at index 0
			//	//converted.emplace_back(std::move(genes), 1000, individual.getFeatures()); //C is at index 0
			//}

			////convert SV pool to proper indexing
			//std::vector<Gene> svVectorsWithGammas;
			//for (auto i = 0u; i < m_svFrozenPool.size(); ++i)
			//{
			//	svVectorsWithGammas.emplace_back(m_trainingSet.size() + i, m_svFrozenPool[i].classValue, m_svFrozenPool[i].gamma);
			//}

			//std::vector<DatasetVector> svPoolIds;
			//for (auto i = 0u; i < m_svFrozenPool.size(); ++i)
			//{
			//	svPoolIds.emplace_back(m_svFrozenPool[i].id, m_svFrozenPool[i].classValue);
			//}

			//svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(svPoolIds) };
			//auto m_joined_T_V = joinSets(m_fullDatasetWorkflow.getTraningSet(), m_fullDatasetWorkflow.getValidationSet());
			//auto svPoolSet = uncertainTraining.convertChromosome(m_joined_T_V);
			//auto train_and_svPool = joinSets(m_trainingSet, svPoolSet);

			//Population<SvmCustomKernelChromosome> populationConverted{ converted };

			////frozen pool indexing need full dataset ids, chromosomes have ids from uncertain region only, proper indexing is kept in ensembleTreeWorkflow with all of the pools
			////dasvmTraining.trainPopulation(populationConverted, m_trainingSet, m_svFrozenPool);
			//dasvmTraining.trainPopulation(populationConverted, train_and_svPool, svVectorsWithGammas);

			////get back svm to current pop in order to evaluate fitness
			//for (auto i = 0u; i < populationConverted.size(); ++i)
			//{
			//	pop[i].updateClassifier(populationConverted[i].getClassifier());
			//}




			svmComponents::SvmTrainingCustomKernel dasvmTraining(m_algorithmConfig.m_svmConfig, false, "RBF_MAX", true);



			for (auto& individual : pop)
			{
				std::vector<SvmCustomKernelChromosome> converted;

				std::vector<Gene> genes;
				auto gamma = individual.getKernelParameters()[1]; //gamma is at index 1
				auto trainingSet = individual.getTraining().getDataset();
				for (auto tr : trainingSet)
				{
					genes.emplace_back(tr.id, tr.classValue, gamma); //convert training set
				}

				converted.emplace_back(std::move(genes), individual.getKernelParameters()[0], individual.getFeatures()); //C is at index 0

				//convert SV pool to proper indexing
				std::vector<Gene> svVectorsWithGammas;
				for (auto i = 0u; i < m_svFrozenPool.size(); ++i)
				{
					svVectorsWithGammas.emplace_back(m_trainingSet.size() + i, m_svFrozenPool[i].classValue, gamma);
				}

				std::vector<DatasetVector> svPoolIds;
				for (auto i = 0u; i < m_svFrozenPool.size(); ++i)
				{
					svPoolIds.emplace_back(m_svFrozenPool[i].id, m_svFrozenPool[i].classValue);
				}


				svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(svPoolIds) };
				auto m_joined_T_V = joinSets(m_fullDatasetWorkflow.getTraningSet(), m_fullDatasetWorkflow.getValidationSet());
				auto svPoolSet = uncertainTraining.convertChromosome(m_joined_T_V);
				auto train_and_svPool = joinSets(m_trainingSet, svPoolSet);

				Population<SvmCustomKernelChromosome> populationConverted{ converted };

				//frozen pool indexing need full dataset ids, chromosomes have ids from uncertain region only, proper indexing is kept in ensembleTreeWorkflow with all of the pools
				//dasvmTraining.trainPopulation(populationConverted, m_trainingSet, m_svFrozenPool);
				dasvmTraining.trainPopulation(populationConverted, train_and_svPool, svVectorsWithGammas);

				//get back svm to current pop in order to evaluate fitness

				individual.updateClassifier(populationConverted[0].getClassifier());
			}

		}
		else if (m_useDasvmKernel)
		{
			//TODO parametrize this
			svmComponents::SvmTrainingCustomKernel dasvmTraining(m_algorithmConfig.m_svmConfig, false, "RBF_MAX", true);

			std::vector<SvmCustomKernelChromosome> converted;

			for (auto individual : pop)
			{
				std::vector<Gene> genes;
				auto gamma = individual.getKernelParameters()[1]; //gamma is at index 1
				auto trainingSet = individual.getTraining().getDataset();
				for (auto tr : trainingSet)
				{
					genes.emplace_back(tr.id, tr.classValue, gamma); //convert training set
				}

				converted.emplace_back(std::move(genes), individual.getKernelParameters()[0], individual.getFeatures()); //C is at index 0
			}

			//convert SV pool to proper indexing
			std::vector<Gene> svVectorsWithGammas;
			for (auto i = 0u; i < m_svFrozenPool.size(); ++i)
			{
				svVectorsWithGammas.emplace_back(m_trainingSet.size() + i, m_svFrozenPool[i].classValue, m_svFrozenPool[i].gamma);
			}

			std::vector<DatasetVector> svPoolIds;
			for (auto i = 0u; i < m_svFrozenPool.size(); ++i)
			{
				svPoolIds.emplace_back(m_svFrozenPool[i].id, m_svFrozenPool[i].classValue);
			}

			svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(svPoolIds) };
			auto svPoolSet = uncertainTraining.convertChromosome(m_fullDatasetWorkflow.getTraningSet());
			auto train_and_svPool = joinSets(m_trainingSet, svPoolSet);

			Population<SvmCustomKernelChromosome> populationConverted{ converted };

			//frozen pool indexing need full dataset ids, chromosomes have ids from uncertain region only, proper indexing is kept in ensembleTreeWorkflow with all of the pools
			//dasvmTraining.trainPopulation(populationConverted, m_trainingSet, m_svFrozenPool);
			dasvmTraining.trainPopulation(populationConverted, train_and_svPool, svVectorsWithGammas);

			//get back svm to current pop in order to evaluate fitness
			for (auto i = 0u; i < populationConverted.size(); ++i)
			{
				pop[i].updateClassifier(populationConverted[i].getClassifier());
			}
		}
		else
		{
			m_svmTraining.launch(pop, m_trainingSet);
		}
	}



	void BigSetsSvmHelper::evaluate(Population<SvmSimultaneousChromosome>& pop)
	{
		dataset::Dataset<std::vector<float>, float> dataset;

		//Full V variant 02.05.2022
		//dataset = joinSets(m_trainingSet, m_fullDatasetWorkflow.getValidationSet());
		
		if (m_newDatasetFlow)
		{
			//dataset = joinSets(m_trainingSet, m_validationSet);
			dataset = m_validationSet;
		}
		else
		{
			dataset = m_validationSet;
			//dataset = joinSets(m_trainingSet, m_validationSet);
			//dataset = joinSets(m_trainingSet, m_fullDatasetWorkflow.getValidationSet());  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!this was the correct variant uncomment after experiment
			//dataset = joinSets(m_fullDatasetWorkflow.getTraningSet(), m_fullDatasetWorkflow.getValidationSet());	
		}



		//auto setThresholds = dynamic_cast<SvmHyperplaneDistance*>(m_estimationMethod.get());
		//const size_t iterationCount = std::distance(pop.begin(), pop.end());
		//auto first = pop.begin();

		//#pragma omp parallel for
		/*for (int i = 0; i < static_cast<int>(iterationCount); i++)
		{
			auto& individual = *(first + i);

			if(m_useFeatureSelection)
			{
				auto individualDataset = individual.convertFeatures(dataset);
				setThresholds->calculateThresholds(individual, individualDataset);
			}
			else
			{
				setThresholds->calculateThresholds(individual, dataset);
			}

		}*/

		m_validationTest->launch(pop, dataset);
		//m_validationTest->launch(m_popTestSet, dataset);
		m_popTestSet = pop;  //TODO fix test pop in here as it seems to be wrong (although it is never used)

		if (m_debugLog)
		{
			std::ofstream poplog(m_config.outputFolderPath.string() + "PopulationFitnessDistanceLog__" + std::to_string(m_nodeNumber) + ".txt", std::ios_base::app);
			auto separator = ';';
			for (auto i = 0; i < pop.size(); ++i)
			{
				poplog << m_generationNumber << separator << pop[i].getMetric().m_fitness << separator << pop[i].getMetric().m_additionalValue << separator <<
					m_trainingSetOptimization->getCurrentTrainingSetSize() << "\n";
			}
			poplog.close();
		}
		log();
	}

	bool BigSetsSvmHelper::isFinished()
	{
		return m_stopConditionElement.launch(m_pop);
	}

	void BigSetsSvmHelper::log()
	{
		//TODO add header to logs and modify python analysis to use this header
		auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
		//auto featureNumber = m_validationSet.getSamples()[0].size();
		auto bestOneIndex = m_pop.getBestIndividualIndex();

		m_resultLogger.createLogEntry(m_pop,
			m_popTestSet,
			m_timer,
			m_algorithmName,
			m_generationNumber,
			Accuracy(bestOneConfustionMatrix),
			m_pop.getBestOne().featureSetSize(),
			0, //TODO size of training size
			bestOneConfustionMatrix,
			m_popTestSet[bestOneIndex].getConfusionMatrix().value());
	}

	void BigSetsSvmHelper::switchMetric()
	{
		LOG_F(WARNING, "switchMetric in SvmEnsmebleHelper.cpp is deprecated");
		m_estimationMethod = std::make_shared<SvmHyperplaneDistance>(true, true);
		m_validation = std::make_shared<svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome>>(*m_estimationMethod, false);
		m_validationTest = std::make_shared<svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome>>(*m_estimationMethod, true);
	}
} // namespace genetic
