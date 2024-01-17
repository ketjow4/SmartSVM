#include <set>
#include <libStrategies/FileSinkStrategy.h>
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmStrategies/GridSearchPopulationGenerationStrategy.h"
#include "libSvmStrategies/GridSearchCrossValidationStrategy.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "GridSearchWorkflow.h"
#include "WorkflowUtils.h"
#include "libSvmComponents/GaSvmGeneration.h"
#include "libRandom/MersenneTwister64Rng.h"
#include "libSvmComponents/SvmUtils.h"
#include "libPlatform/loguru.hpp"
#include "SvmLib/EnumsTranslations.h"
#include "libSvmComponents/SvmHyperplaneDistance.h"
#include "libPlatform/TimeUtils.h"

namespace genetic
{
using namespace svmComponents;
using namespace geneticComponents;

GridSearchWorkflow::GridSearchWorkflow(const SvmWokrflowConfiguration& config,
                                       GridSearchConfiguration&& algorithmConfig,
                                       IDatasetLoader& workflow)
    : m_algorithmConfig(std::move(algorithmConfig))
    , m_valdiationElement(std::make_shared<svmStrategies::SvmValidationStrategy<SvmKernelChromosome>>(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false))
    , m_valdiationTestDataElement(std::make_shared<svmStrategies::SvmValidationStrategy<SvmKernelChromosome>>(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true))
    , m_gridSearchCrossValidationElement(m_algorithmConfig)
    , m_createGridPopulationElement(*m_algorithmConfig.m_kernel)
    , m_savePngElement()
    , m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
    , m_loadingWorkflow(workflow)
    , m_config(config)
{
    m_trainingSet = &m_loadingWorkflow.getTraningSet();
    m_validationSet = &m_loadingWorkflow.getValidationSet();
    m_testSet = &m_loadingWorkflow.getTestSet();
}




void log(const std::string& filename, const SvmWokrflowConfiguration& config, const Population<SvmKernelChromosome>& population)
{
	std::filesystem::path logFilePath(config.outputFolderPath.string() + timeUtils::getTimestamp() + filename + ".txt");

	std::ofstream logFile(logFilePath);

	auto kernelType = population.getBestOne().getKernelType();
	
	if (kernelType == phd::svm::KernelTypes::Poly)
	{
		logFile << "Polynomial kernel:\n";
		logFile << "# C \t Degree \t Fitness\n";
	}
	else if (kernelType == phd::svm::KernelTypes::Linear)
	{
		logFile << "Linear kernel:\n";
		logFile << "# C  \t Fitness\n";
	}
	else if (kernelType == phd::svm::KernelTypes::RBF_POLY_GLOBAL)
	{
		logFile << "RBF_POLY_GLOBAL kernel:\n";
		logFile << "# C  \t gamma \t t \t degree \t Fitness\n";
	}
	else
	{
		logFile << "# C \t Gamma \t Fitness\n";
	}
	

	for (auto& individual : population)
	{
		//C gamma
		if(kernelType == phd::svm::KernelTypes::Poly)
		{
			logFile << individual.getKernelParameters()[0] << "\t" << individual.getKernelParameters()[1] << "\t" << individual.getFitness() << "\n";
		}
		else if(kernelType == phd::svm::KernelTypes::Linear)
		{
			logFile << individual.getKernelParameters()[0] << "\t" << individual.getFitness() << "\n";
		}
		else if (kernelType == phd::svm::KernelTypes::RBF_POLY_GLOBAL)
		{
			logFile << individual.getKernelParameters()[0] << "\t"  << individual.getKernelParameters()[1] << "\t"
			<< individual.getKernelParameters()[2] << "\t" << individual.getKernelParameters()[3] << "\t" << individual.getFitness() << "\n";
		}
		else
		{
			logFile << individual.getKernelParameters()[0] << "\t" << individual.getKernelParameters()[1] << "\t" << individual.getFitness() << "\n";
		}
	}
}

std::shared_ptr<phd::svm::ISvm> GridSearchWorkflow::runWithGeneration(geneticComponents::IPopulationGeneration<svmComponents::SvmTrainingSetChromosome>& populationGeneration)
{
	try
	{
		auto trainingSets = populationGeneration.createPopulation(m_algorithmConfig.m_subsetIterations);

		for (const auto& trainingSet : trainingSets)
		{
			m_population = geneticComponents::Population<SvmKernelChromosome>();

			auto trainingSubset = trainingSet.convertChromosome(*m_trainingSet);

			doGridSearch(trainingSubset);
		}
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what() << "\n";
	}

	auto population = m_population;
	return (population.getBestOne().getClassifier());
}

void GridSearchWorkflow::switchMetric(std::shared_ptr<svmComponents::ISvmMetricsCalculator> metric)
{
	m_valdiationElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmKernelChromosome>>(*metric, false);
	m_valdiationTestDataElement = std::make_shared<svmStrategies::SvmValidationStrategy<SvmKernelChromosome>>(*metric, true);
}

void GridSearchWorkflow::internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmKernelChromosome> pop)
{
	for(const auto& individual : pop)
	{
		//if (individual.getFitness() > 0.6)
		{
			//filesystem::FileSystem fs;
			auto detailsPath = m_config.outputFolderPath.string() + "\\details\\";
			std::filesystem::create_directories(detailsPath);
			auto out = std::filesystem::path(detailsPath + timeUtils::getTimestamp() +
				"__gamma__" + std::to_string(individual.getClassifier()->getGamma()) +
				"__C__" + std::to_string(individual.getClassifier()->getC()) + "__iter__" +
				std::to_string(0) + 
				"_acc_" + std::to_string(individual.getConfusionMatrix().value().accuracy()) + "__" +
				".png");

			if (individual.getKernelType() == phd::svm::KernelTypes::Linear)
			{
				out = std::filesystem::path(detailsPath + timeUtils::getTimestamp() + "_LINEAR_"
					"__C__" + std::to_string(individual.getClassifier()->getC()) + "__iter__" +
					std::to_string(0) +
					"_acc_" + std::to_string(individual.getConfusionMatrix().value().accuracy()) + "__" + ".png");
			}
			else if (individual.getKernelType() == phd::svm::KernelTypes::Poly)
			{
				out = std::filesystem::path(detailsPath + timeUtils::getTimestamp() + "_POLY_"
					"__degree__" + std::to_string(individual.getClassifier()->getDegree()) +
					"__C__" + std::to_string(individual.getClassifier()->getC()) + "__iter__" + 
					std::to_string(0) +
					"_acc_" + std::to_string(individual.getConfusionMatrix().value().accuracy()) + "__" + ".png");
			}

			auto svm = individual.getClassifier();
			SvmVisualization visualization;
			auto m_image = visualization.createDetailedVisualization(*svm,
			                                                         m_algorithmConfig.m_svmConfig.m_height,
			                                                         m_algorithmConfig.m_svmConfig.m_width,
			                                                         *m_trainingSet, *m_validationSet, *m_testSet);
			//auto img =  gsl::make_span(m_image);

			m_savePngElement.launch(m_image, out);
		}
	}
}

void GridSearchWorkflow::getValuesForCustomKernel(geneticComponents::Population<svmComponents::SvmKernelChromosome> populationValidationFitness)
{
	std::sort(populationValidationFitness.begin(), populationValidationFitness.end(), [](SvmKernelChromosome a, SvmKernelChromosome b)
	{
			return a.getFitness() > b.getFitness();
	});

	auto i = 0;

	std::set<double> goodGammas;
	std::set<double> goodC;
	for (auto& individual : populationValidationFitness)
	{
		if (i < 4)
		{
			goodGammas.insert(individual.getClassifier()->getGamma());
			goodC.insert(individual.getClassifier()->getC());
		}
		++i;
	}
	m_goodGammas.assign(goodGammas.begin(), goodGammas.end());
	m_goodC.assign(goodC.begin(), goodC.end());

	//auto bestMinusThreshold = populationValidationFitness.getBestOne().getFitness() - 0.01; //magic threshold

	//std::set<double> goodGammas;
	//for(auto& individual : populationValidationFitness)
	//{
	//	if( individual.getFitness() > bestMinusThreshold)
	//	{
	//		goodGammas.insert(individual.getClassifier()->getGamma());
	//	}
	//}
	//m_goodGammas.assign(goodGammas.begin(), goodGammas.end());

	//std::set<double> goodC;
	//for (auto& individual : populationValidationFitness)
	//{
	//	if (individual.getFitness() > bestMinusThreshold)
	//	{
	//		goodC.insert(individual.getClassifier()->getC());
	//	}
	//}
	//m_goodC.assign(goodC.begin(), goodC.end());
}

void GridSearchWorkflow::doGridSearch(const dataset::Dataset<std::vector<float>, float>& trainingSubset)
{
	for (unsigned int iteration = 0; iteration < m_algorithmConfig.m_numberOfIterations; iteration++)
	{
		auto pop = m_createGridPopulationElement.launch(m_population);

		auto pop2 = m_gridSearchCrossValidationElement.launch(pop, trainingSubset);
		auto populationValidationFitness = pop2;

		if (auto est = dynamic_cast<SvmHyperplaneDistance*>(m_algorithmConfig.m_svmConfig.m_estimationMethod.get()); est != nullptr)
		{
			for(auto& p : populationValidationFitness)
			{
				est->calculateThresholds(p, *m_validationSet);
			}
		}
		
		m_population = m_valdiationElement->launch(populationValidationFitness, *m_validationSet);
		pop2 = m_population;
		auto populationWithTestDataFitness = m_valdiationTestDataElement->launch(pop2, *m_testSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, iteration);
			auto image = m_createVisualizationElement.launch(m_population, *m_trainingSet, *m_validationSet);
			m_savePngElement.launch(image, m_pngNameSource);

			svmComponents::SvmVisualization visualization2;
			auto temp = m_population.getBestOne().getClassifier();
			auto image2 = visualization2.createDetailedVisualization(*temp, 500, 500, *m_trainingSet, *m_validationSet, *m_testSet);
			SvmWokrflowConfiguration config_copy2{ "","","",m_config.outputFolderPath, "GridSearch_","" };
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, 0);
			m_savePngElement.launch(image2, m_pngNameSource);

			/*svmComponents::SvmVisualization visualization3;
			auto temp2 = populationWithTestDataFitness[m_population.getBestIndividualIndex()].getClassifier();
			auto image3 = visualization3.createDetailedVisualization(*temp2, 500, 500, *m_trainingSet, *m_validationSet, *m_testSet);
			SvmWokrflowConfiguration config_copy3{ "","","",m_config.outputFolderPath, "GridSearch_TestSetThreshold","" };
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, 0);
			m_savePngElement.launch(image3, m_pngNameSource);*/

			//internalVisualization_Debugging(pop);
		}

		auto bestOneConfustionMatrix = m_population.getBestOne().getConfusionMatrix().value();
		auto bestOneIndex = m_population.getBestIndividualIndex();
		auto validationDataset = *m_validationSet;
		auto featureNumber = validationDataset.getSamples()[0].size();
		auto trainingSetSize = trainingSubset.size();

		std::sort(populationValidationFitness.begin(), populationValidationFitness.end(), [](SvmKernelChromosome a, SvmKernelChromosome b)
			{
				return a.getFitness() > b.getFitness();
			});


		// if(m_config.verbosity == testApp::Verbosity::Standard || m_config.verbosity == testApp::Verbosity::All)
		// {
		// 	log("gridSearchFitnessLog_subset" + std::to_string(m_algorithmConfig.m_subsetSize), m_config, populationValidationFitness);
		// }
		
		//used these results in initialization of custom gammas
		getValuesForCustomKernel(populationValidationFitness);

		m_resultLogger.createLogEntry(m_population,
		                              populationWithTestDataFitness,
		                              m_timer,
		                              std::string( kernelTypeToString(m_algorithmConfig.m_svmConfig.m_kernelType).data()),
		                              iteration,
		                              Accuracy(bestOneConfustionMatrix),
		                              featureNumber,
		                              trainingSetSize,
		                              bestOneConfustionMatrix,
								      populationWithTestDataFitness[bestOneIndex].getConfusionMatrix().value());
	}
}



unsigned int getNumberOfClassExamples2(unsigned int numberOfClassExamples, std::vector<unsigned int> labelsCount, double thresholdForMaxNumberOfClassExamples)
{
	auto minorityClassExamplesNumber = static_cast<unsigned int>(*std::min_element(labelsCount.begin(), labelsCount.end()) * thresholdForMaxNumberOfClassExamples);
	if (minorityClassExamplesNumber < numberOfClassExamples)
		return minorityClassExamplesNumber;
	return numberOfClassExamples;
}

void GridSearchWorkflow::runGridSearch()
{
    try
    {
	    if (m_algorithmConfig.m_subsetSize == 0)
		{
			doGridSearch(*m_trainingSet);
		}
		else
		{
			auto labels = m_trainingSet->getLabels();
			auto numberOfClasses = std::set<float>(labels.begin(), labels.end()).size();

			auto subsetSize = getNumberOfClassExamples2(m_algorithmConfig.m_subsetSize, svmUtils::countLabels(static_cast<unsigned int>(numberOfClasses), *m_trainingSet), 1.0);
			
			GaSvmGeneration generation{
				*m_trainingSet,
				std::make_unique<random::MersenneTwister64Rng>(std::chrono::system_clock::now().time_since_epoch().count()),
				subsetSize,
				svmUtils::countLabels(static_cast<unsigned int>(numberOfClasses), *m_trainingSet)
			};

			auto trainingSets = generation.createPopulation(m_algorithmConfig.m_subsetIterations);

			for (const auto& trainingSet : trainingSets)
			{
				m_population = geneticComponents::Population<SvmKernelChromosome>();

				auto trainingSubset = trainingSet.convertChromosome(*m_trainingSet);

				doGridSearch(trainingSubset);
			}
		}
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what() << "\n";
    }
}

std::shared_ptr<phd::svm::ISvm> GridSearchWorkflow::run()
{
    try
    {
        runGridSearch();
        m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));
    }
    catch (...)
    {
		LOG_F(ERROR, "Unknown error in GridSearch run.");
		std::cout << "Unknown error in GridSearch run." << "\n";
    }
    auto population = m_population;
    return (population.getBestOne().getClassifier());
}
} // namespace genetic
