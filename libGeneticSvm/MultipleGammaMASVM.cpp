#include "MultipleGammaMASVM.h"
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
	MultipleGammaMASVMWorkflow::MultipleGammaMASVMWorkflow(const SvmWokrflowConfiguration& config,
		svmComponents::MutlipleGammaMASVMConfig algorithmConfig,
		IDatasetLoader& workflow)
		: m_config(config)
		, m_loadingWorkflow(workflow)
		, m_algorithmConfig(std::move(algorithmConfig))
		, m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
		, m_valdiationElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
		, m_valdiationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
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
		, m_validationSuperIndividualsElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
		, m_parentSelectionElement(*m_algorithmConfig.m_parentSelection)
		, m_compensationGenerationElement(std::move(m_algorithmConfig.m_compensationGenerationElement))
		, m_trainingSet(nullptr)
		, m_validationSet(nullptr)
		, m_testSet(nullptr)
		, m_generationNumber(0)
		, m_numberOfClassExamples(m_algorithmConfig.m_numberOfClassExamples)
		//, m_initialNumberOfClassExamples(m_algorithmConfig.m_numberOfClassExamples)
	{
		m_shrinkTrainingSet = false;
	}

	std::shared_ptr<phd::svm::ISvm> MultipleGammaMASVMWorkflow::run()
	{
		initializeGeneticAlgorithm();
		runGeneticAlgorithm();
		m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));

		return m_population.getBestOne().getClassifier();
	}

	void MultipleGammaMASVMWorkflow::initializeGeneticAlgorithm()
	{
		if (m_trainingSet == nullptr)
		{
			m_trainingSet = &m_loadingWorkflow.getTraningSet();
			m_validationSet = &m_loadingWorkflow.getValidationSet();
			m_testSet = &m_loadingWorkflow.getTestSet();
		}

		GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
			1,
			1,
			m_numberOfClassExamples,
			1,
			std::make_shared<svmComponents::SvmKernelTraining>(
				m_algorithmConfig.m_svmConfig,
				m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::
				Auc),
			std::make_shared<svmComponents::RbfKernel>(cv::ml::ParamGrid(0.001, 1050, 10),
				cv::ml::ParamGrid(0.001, 1050, 10),
				false)),
			m_loadingWorkflow);

		auto bestOne = gs.run();

		try
		{
			//gs.getC();


			//m_gammaRange = gs.getGammas();
			m_gammaRange = { bestOne->getGamma() / 10, bestOne->getGamma(), bestOne->getGamma() * 10, bestOne->getGamma() * 100, bestOne->getGamma() * 1000 };
			m_CValue = bestOne->getC();


			//set all parameters in here!!!!
			//std::vector<double> gammaTest = {/*0.1,*/ 10, 500, 1000 };
			//m_gammaRange = gammaTest;
			//m_CValue = 1000; 



		}
		catch (const std::exception& exception)
		{
			LOG_F(ERROR, "Error: %s", exception.what());
			std::cout << exception.what();
			//m_logger.LOG(logger::LogLevel::Error, exception.what());
		}
	}

	void MultipleGammaMASVMWorkflow::runGeneticAlgorithm()
	{
		//reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGenerationSequential>&>(m_algorithmConfig.m_populationGeneration)->setNumberOfClassExamples(m_numberOfClassExamples);
		//m_adaptationElement.resetToInitial(m_numberOfClassExamples);
		//m_currentGamma = m_gammaRange[i];

		m_superIndividualsGenerationElement->setC(m_CValue);
		reinterpret_cast<const std::shared_ptr<svmComponents::MultipleGammaGeneration>&>(m_algorithmConfig.m_populationGeneration)->setCandGamma(m_CValue, m_gammaRange);
		reinterpret_cast<const std::shared_ptr<svmComponents::MultipleGammaMutation>&>(m_algorithmConfig.m_mutation)->setGamma(m_gammaRange);
		m_crossoverCompensationElement.setGamma(m_gammaRange);
		//m_supportVectorPoolElement.setCurrentGamma(m_gammaRange[i]);

		
		initMemetic();
		memeticAlgorithm();

		//visualize the best vectors - train the best one and plot the results
		geneticComponents::Population<SvmCustomKernelChromosome> best_pop;
		//visualizeFrozenSet(best_pop);
	}

	void MultipleGammaMASVMWorkflow::logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
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
			m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
			bestOneConfustionMatrix,
			testPopulation[bestOneIndex].getConfusionMatrix().value());
		m_generationNumber++;
	}

	void MultipleGammaMASVMWorkflow::initMemetic()
	{
		try
		{
			m_svPool.clear();
			m_supportVectorPoolElement.clear();
			//m_numberOfClassExamples = m_initialNumberOfClassExamples;
			//m_adaptationElement.resetToInitial(m_initialNumberOfClassExamples);
			m_adaptationElement.setFrozenSetSize(static_cast<unsigned int>(m_frozenSV.size()));

			auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);

			auto tr = reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernel>&>(m_algorithmConfig.m_training);
			tr->trainPopulation(population, *m_trainingSet, m_frozenSV);

			m_population = m_valdiationElement.launch(population, *m_validationSet);

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
					SvmWokrflowConfiguration config_copy2{ "","","",m_config.outputFolderPath, "max_iter_Problem","" };
					setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, m_generationNumber);
					m_savePngElement.launch(image2, m_pngNameSource);
				}
			}

			auto testPopulation = m_valdiationTestDataElement.launch(population, *m_testSet);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);

				auto svm = m_population.getBestOne().getClassifier();
				svmComponents::SvmVisualization visualization;
				//auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
				auto best = m_population.getBestOne();

				visualization.setGene(best);
				//auto[map, scores] = res2->check_sv(*m_trainingSet);
				//auto[map, scores] = res2->check_sv(*m_validationSet);
				//visualization.setScores(scores);
				//visualization.setMap(map);
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
		catch (const std::exception& exception)
		{
			LOG_F(ERROR, "Error: %s", exception.what());
			std::cout << exception.what();
			throw;
			//m_logger.LOG(logger::LogLevel::Error, std::string("Unknown exception: ") + exception.what());
		}
	}

	void MultipleGammaMASVMWorkflow::memeticAlgorithm()
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
				auto afterValidtion = m_valdiationElement.launch(populationEducated, *m_validationSet);

				m_supportVectorPoolElement.updateSupportVectorPool(afterValidtion, *m_trainingSet);
				m_svPool = m_supportVectorPoolElement.getSupportVectorPool();

				auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
				auto superIndividualsPopulation = m_superIndividualsGenerationElement->createPopulation(superIndividualsSize, m_svPool, m_numberOfClassExamples);


				tr->trainPopulation(superIndividualsPopulation, *m_trainingSet, m_frozenSV);
				m_validationSuperIndividualsElement.launch(superIndividualsPopulation, *m_validationSet);

				auto combinedPopulation = m_populationCombinationElement.launch(afterValidtion, superIndividualsPopulation);
				m_population = m_selectionElement.launch(m_population, combinedPopulation);

				/*auto [IsModeLocal, NumberOfClassExamples] = */
				m_adaptationElement.adapt(m_population);
				auto IsModeLocal = m_adaptationElement.getIsModeLocal();
				auto NumberOfClassExamples = m_adaptationElement.getNumberOfClassExamples();
				m_numberOfClassExamples = NumberOfClassExamples;

				auto copy = m_population;
				auto testPopulation = m_valdiationTestDataElement.launch(copy, *m_testSet);

				isStop = m_stopConditionElement.launch(m_population);

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
						*m_trainingSet, *m_validationSet, *m_testSet);

					m_savePngElement.launch(image, m_pngNameSource);
				}

				auto localGlobal = dynamic_cast<geneticComponents::LocalGlobalAdaptationSelection<SvmCustomKernelChromosome>*>(m_algorithmConfig
					.m_parentSelection.get());
				if (localGlobal != nullptr)
				{
					localGlobal->setMode(IsModeLocal);
				}
				//logAllModels(testPopulation);
				logResults(m_population, testPopulation);
			}
		}
		catch (const std::exception& exception)
		{
			LOG_F(ERROR, "Error: %s", exception.what());
			std::cout << exception.what();
			throw;
		}
	}
} // namespace genetic
