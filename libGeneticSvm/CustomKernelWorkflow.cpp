//#include "libStrategies/TabularDataProviderStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmComponents/SvmTraining.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "WorkflowUtils.h"
#include "SvmExceptions.h"

#include "CustomKernelWorkflow.h"
#include "GridSearchWorkflow.h"
#include "libSvmComponents/RbfKernel.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/CustomKernelTraining.h"
#include "libPlatform/loguru.hpp"
#include "libPlatform/TimeUtils.h"

namespace genetic
{
CustomKernelWorkflow::CustomKernelWorkflow(const SvmWokrflowConfiguration& config,
                                           svmComponents::CustomKernelEvolutionConfiguration algorithmConfig,
                                           IDatasetLoader& workflow)
	: m_algorithmConfig(std::move(algorithmConfig))
	, m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
	, m_validationElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
	, m_validationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
	, m_stopConditionElement(*m_algorithmConfig.m_stopCondition)
	, m_crossoverElement(*m_algorithmConfig.m_crossover)
	, m_mutationElement(*m_algorithmConfig.m_mutation)
	, m_selectionElement(*m_algorithmConfig.m_selection)
	, m_createPopulationElement(*m_algorithmConfig.m_populationGeneration)
	, m_savePngElement()
	, m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
	, m_crossoverParentSelectionElement(*m_algorithmConfig.m_parentSelection)
	, m_trainingSet(nullptr)
	, m_validationSet(nullptr)
	, m_testSet(nullptr)
	, m_loadingWorkflow(workflow)
	, m_generationNumber(0)
	, m_config(config)
{
}

std::shared_ptr<phd::svm::ISvm> CustomKernelWorkflow::run()
{
	initializeGeneticAlgorithm();
	runGeneticAlgorithm();
	m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));
	m_region1Logger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename + "__REGION_1.txt"));
	m_region2Logger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename + "__REGION_2.txt"));

	return m_population.getBestOne().getClassifier();
}

void CustomKernelWorkflow::customLogTrainingSetAndGammas(std::shared_ptr<phd::svm::ISvm> svm)
{
	std::ofstream out(m_pngNameSource.string() + timeUtils::getTimestamp() + ".csv");
	out << "AUC: " << m_population.getBestOne().getFitness() << std::endl;
	out << "Tr: ";
	for (auto v : m_population.getBestOne().getDataset())
		out << v.id << " ";
	out << std::endl;

	out << "Gammas: ";
	for (auto v : m_population.getBestOne().getDataset())
		out << v.gamma << " ";
	out << std::endl;

	auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
	out << "Alphas: ";
	for (auto i = 0; i < res->m_model->l; ++i)
	{
		out << res->m_model->sv_coef[0][i] << " ";
	}
	out << std::endl;
}

std::vector<svmComponents::Gene> m_supportVectorPool;


unsigned int findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
                                         gsl::span<const float> supportVector)
{
	auto samples = individualDataset.getSamples();
	auto positionInDataset = std::find_if(samples.begin(),
	                                      samples.end(),
	                                      [&supportVector](const auto& sample)
	                                      {
		                                      return std::equal(sample.begin(),
		                                                        sample.end(),
		                                                        supportVector.begin(),
		                                                        supportVector.end());
	                                      }) - samples.begin();
	return static_cast<unsigned int>(positionInDataset);
}

std::vector<svmComponents::Gene> getBestSupportVector(const svmComponents::SvmCustomKernelChromosome& chromosome,
                                                      const dataset::Dataset<std::vector<float>, float>& validationSet,
                                                      const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                      double scoreThreshold = 1.0) //cannot be bigger than 1.0
{
    const auto classifier = chromosome.getClassifier();
    if (classifier && classifier->isTrained())
    {
        auto svm = reinterpret_cast<phd::svm::libSvmImplementation*>(classifier.get());
        auto[ommit, scores] = svm->check_sv(validationSet);
        auto supportVectors = classifier->getSupportVectors();

        auto labels = trainingSet.getLabels();
        std::vector<svmComponents::Gene> result;

        for (auto i = 0; i < supportVectors.size(); i++)
        {
            if (scores[i] >= scoreThreshold)
            {
                const float* sv = supportVectors[i].data();
                const gsl::span<const float> supportVector(sv, supportVectors[i].size());

                const auto positionInDataset = findPositionOfSupprotVector(trainingSet, supportVector);
                auto gammas = svm->getGammas();
                result.emplace_back(svmComponents::Gene(positionInDataset, labels[positionInDataset], gammas[i]));
            }
        }

        return result;
    }
    throw svmComponents::UntrainedSvmClassifierException();
}

void addSupportVectors(const svmComponents::SvmCustomKernelChromosome& chromosome,
                       const dataset::Dataset<std::vector<float>, float>& validationSet,
                       const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	const auto classifier = chromosome.getClassifier();
	if (classifier && classifier->isTrained())
	{
		auto svm = reinterpret_cast<phd::svm::libSvmImplementation*>(classifier.get());
		auto [ommit, scores] = svm->check_sv(validationSet);
		auto supportVectors = classifier->getSupportVectors();

		auto labels = trainingSet.getLabels();

		for (auto i = 0; i < supportVectors.size(); i++)
		{
			if (scores[i] > 0.95)
			{
				const float* sv = supportVectors[i].data();
				const gsl::span<const float> supportVector(sv, supportVectors[i].size());

				const auto positionInDataset = findPositionOfSupprotVector(trainingSet, supportVector);

				//if (m_supportVectorIds.emplace(positionInDataset).second)
				//{
				auto gammas = svm->getGammas();
				m_supportVectorPool.emplace_back(svmComponents::Gene(positionInDataset, labels[positionInDataset], gammas[i]));
				//}
			}
		}

		return;
	}
	//throw UntrainedSvmClassifierException();
}

void updatePool(const geneticComponents::Population<svmComponents::SvmCustomKernelChromosome>& population,
                const dataset::Dataset<std::vector<float>, float>& validationSet,
	const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	for (const auto& individual : population)
	{
		addSupportVectors(individual, validationSet, trainingSet);
	}
}

void CustomKernelWorkflow::internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop)
{
	for (const auto& individual : pop)
	{
		//if (individual.getFitness() > 0.6)
		{
			//filesystem::FileSystem fs;
			auto detailsPath = m_config.outputFolderPath.string() + "\\details\\";
			std::filesystem::create_directories(detailsPath);
			auto out = std::filesystem::path(detailsPath + timeUtils::getTimestamp() +
				+ "__iter__" +
				std::to_string(0) + ".png");

			auto svm = individual.getClassifier();
			svmComponents::SvmVisualization visualization;

			auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
			auto[map, scores] = res2->check_sv(*m_validationSet);
			visualization.setScores(scores);
			visualization.setMap(map);

			auto m_image = visualization.createDetailedVisualization(*svm,
				m_algorithmConfig.m_svmConfig.m_height,
				m_algorithmConfig.m_svmConfig.m_width,
				*m_trainingSet, *m_validationSet, *m_testSet);
			//auto img =  gsl::make_span(m_image);

			m_savePngElement.launch(m_image, out);
		}
	}
}

void CustomKernelWorkflow::runGeneticAlgorithm()
{
	try
	{
		dataset::Dataset<std::vector<float>, float> new_Training;
		std::vector<svmComponents::Gene> bestVectors;
		//for (int i = 0; i < 2; ++i)
		//{
		//	dataset::Dataset<std::vector<float>, float> temp;
		//	if(!new_Training.empty())
		//	{
		//		
		//		temp = new_Training;
		//		for (auto& gene : bestVectors)
		//		{
		//			temp.addSample(m_trainingSet->getSamples()[gene.id], m_trainingSet->getLabels()[gene.id]);
		//			gene.id = temp.size() - 1;
		//		}
		//		reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernel>&>(m_algorithmConfig.m_training)->trainPopulation(m_population, temp, bestVectors);
		//		m_validationElement.launch(m_population, *m_validationSet);
		//	}
			bool isStop = false;
			//for (int j = 0; j < 5; ++j)
			while(!isStop)
			{
				auto parents = m_crossoverParentSelectionElement.launch(m_population);
				auto newPopulation = m_crossoverElement.launch(parents);
				auto populationEducated = m_mutationElement.launch(newPopulation);

				auto tr = reinterpret_cast<const std::shared_ptr<svmComponents::SvmTrainingCustomKernel>&>(m_algorithmConfig.m_training);

				if (!new_Training.empty())
				{
					//tr->trainPopulation(populationEducated, temp, bestVectors);
				}
				else
				{
					tr->trainPopulation(populationEducated, *m_trainingSet, bestVectors);
				}
				auto poptrained = populationEducated;
				auto afterValidtion = m_validationElement.launch(populationEducated, *m_validationSet);

				//internalVisualization_Debugging(populationEducated);

				//sv pool
				updatePool(poptrained, *m_validationSet, *m_trainingSet);

				m_population = m_selectionElement.launch(m_population, afterValidtion);

				auto copy = m_population;
				auto testPopulation = m_validationTestDataElement.launch(copy, *m_testSet);
				isStop = m_stopConditionElement.launch(m_population);

				if (m_algorithmConfig.m_svmConfig.m_doVisualization)
				{
					
					setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
					//auto image = m_createVisualizationElement.launch(m_population, *m_trainingSet, *m_validationSet);
					//m_savePngElement.launch(image, m_pngNameSource);

					auto svm = m_population.getBestOne().getClassifier();
					svmComponents::SvmVisualization visualization;
					auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
					auto best = m_population.getBestOne();

					visualization.setGene(best);
					auto [map, scores] = res2->check_sv(*m_validationSet);
					visualization.setScores(scores);
					visualization.setMap(map);

					if(!new_Training.empty())
					{
					/*	auto image = visualization.createDetailedVisualization(*svm,
							m_algorithmConfig.m_svmConfig.m_height,
							m_algorithmConfig.m_svmConfig.m_width,
							temp, *m_validationSet);

						m_savePngElement.launch(image, m_pngNameSource);*/
					}
					else
					{
						auto image = visualization.createDetailedVisualization(*svm,
							m_algorithmConfig.m_svmConfig.m_height,
							m_algorithmConfig.m_svmConfig.m_width,
							*m_trainingSet, *m_validationSet, *m_testSet);

						m_savePngElement.launch(image, m_pngNameSource);

					}
					customLogTrainingSetAndGammas(svm);
					regionBasedScores();
				}
			
				//logAllModels(testPopulation);
				logResults(m_population, testPopulation);
			}
		/*
		if (i == 1)
			break;

		//std::vector<svmComponents::Gene> chromosomeDataset;
		//std::vector<std::uint64_t> best = { 524,3340,264,1224,4464,856,152,4496 };
		//for (auto d = 0; d < best.size(); ++d)
		//{
		//	chromosomeDataset.emplace_back(svmComponents::Gene(best[d], static_cast<std::uint8_t>(m_trainingSet->getLabels()[best[d]]), 10.0));
		//}

        //bestVectors = getBestSupportVector(m_population.getBestOne(), *m_validationSet, *m_trainingSet, 1.0); //score equal to 1.0
		//auto copy = bestVectors;

        //visualize the best vectors - train the best one and plot the results
		auto copy = m_supportVectorPool;
		bestVectors = m_supportVectorPool;
		
		//svmComponents::SvmCustomKernelChromosome best_vec{ std::move(copy), 1.0 };
		//geneticComponents::Population<SvmCustomKernelChromosome> best_pop{ std::vector<SvmCustomKernelChromosome>{best_vec} };
		//m_trainingSvmClassifierElement.launch(best_pop, *m_trainingSet);

		//svmComponents::SvmVisualization visualization3;
		//auto image3 = visualization3.createDetailedVisualization(*best_pop.getBestOne().getClassifier(), 500, 500, *m_trainingSet, *m_trainingSet);
		//SvmWokrflowConfiguration config_copy3{ "","","",m_config.outputFolderPath, "m_supportVectorPool_individual!!!!!!!!","" };
		//setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, m_generationNumber);
		//m_savePngElement.launch(image3, m_pngNameSource);



		//bestVectors = getBestSupportVector(best_pop.getBestOne(), *m_validationSet, *m_trainingSet, 1.0); //score equal to 1.0
		//bestVectors = chromosomeDataset; //option for debugging


		auto copy2 = bestVectors;
		svmComponents::SvmCustomKernelChromosome best_vec2{ std::move(copy2), 1.0 };
		geneticComponents::Population<SvmCustomKernelChromosome> best_pop2{ std::vector<SvmCustomKernelChromosome>{best_vec2} };
		m_trainingSvmClassifierElement.launch(best_pop2, *m_trainingSet);

		auto samples = m_trainingSet->getSamples();
		auto labels = m_trainingSet->getLabels();
		auto classifier = best_pop2.getBestOne().getClassifier();

		for (auto f = 0; f < samples.size(); ++f)
		{
			if (classifier->classify(samples[f]) == labels[f])
			{
				auto value = fabs(classifier->classifyHyperplaneDistance(samples[f]));
				if(value < 0.3)
				{
					new_Training.addSample(samples[f], labels[f]);
				}
			}
			else
			{
				new_Training.addSample(samples[f], labels[f]);
			}
		}

		svmComponents::SvmVisualization visualization2;
		auto image2 = visualization2.createVisualizationNewTrainingSet(*best_pop2.getBestOne().getClassifier(), 500, 500, new_Training, new_Training);
		SvmWokrflowConfiguration config_copy2{ "","","",m_config.outputFolderPath, "new_training_set_!!!!!!!!","" };
		setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy2, m_generationNumber);
		m_savePngElement.launch(image2, m_pngNameSource);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			SvmWokrflowConfiguration config_copy{ "","","",m_config.outputFolderPath, "support_vectors_pool","" };
	
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy, m_generationNumber);
			
			auto svm = best_pop2.getBestOne().getClassifier();
			svmComponents::SvmVisualization visualization;
			auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
			auto[map, scores] = res2->check_sv(*m_validationSet);
			visualization.setScores(scores);
			visualization.setMap(map);
			auto image = visualization.createDetailedVisualization(*svm,
				m_algorithmConfig.m_svmConfig.m_height,
				m_algorithmConfig.m_svmConfig.m_width,
				*m_trainingSet, *m_validationSet);

			m_savePngElement.launch(image, m_pngNameSource);
		}

		//generacja nowej populacji
	    const auto newGamma = 1000.0;
        reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGauss>&>(m_algorithmConfig.m_mutation)->setGamma(newGamma);
		reinterpret_cast<const std::shared_ptr<svmComponents::MutationCustomGauss>&>(m_algorithmConfig.m_mutation)->setTrainingSet(new_Training);
		auto pop_gen = reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGeneration>&>(m_algorithmConfig.m_populationGeneration);
		pop_gen->setGamma(newGamma);
		pop_gen->setNumberOfClassExamples(4);
		pop_gen->setTrainingSet(new_Training);

		m_population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);*/

		//zamro�enie wektor�w z najlepszego/wybranych z puli itp.
		//}
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
	}
}

void CustomKernelWorkflow::logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
                                      const geneticComponents::Population<SvmCustomKernelChromosome>& testPopulation)
{
	auto bestOneConfustionMatrix = population.getBestOne().getConfusionMatrix().value();
	auto validationDataset = *m_validationSet;
	auto featureNumber = validationDataset.getSamples()[0].size();

	m_resultLogger.createLogEntry(population,
	                              testPopulation,
	                              m_timer,
	                              m_algorithmName,
	                              m_generationNumber,
	                              Accuracy(bestOneConfustionMatrix),
	                              featureNumber,
	                              //m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
	                              bestOneConfustionMatrix);
	m_generationNumber++;
}

void CustomKernelWorkflow::initializeGeneticAlgorithm()
{
	if (m_trainingSet == nullptr)
	{
		m_trainingSet = &m_loadingWorkflow.getTraningSet();
		m_validationSet = &m_loadingWorkflow.getValidationSet();
		m_testSet = &m_loadingWorkflow.getTestSet();
	}

	/*GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	                                                                       1,
	                                                                       1,
	                                                                       16,
	                                                                       1,
	                                                                       std::make_shared<svmComponents::SvmKernelTraining>(
		                                                                       m_algorithmConfig.m_svmConfig,
		                                                                       m_algorithmConfig.m_svmConfig.m_estimationType == svmComponents::svmMetricType::
		                                                                       Auc),
	                                                                       std::make_shared<svmComponents::RbfKernel>(cv::ml::ParamGrid(0.001, 1050, 2),
	                                                                                                                  cv::ml::ParamGrid(0.001, 1050, 2),
	                                                                                                                  false)),
	                      m_loadingWorkflow);

	gs.run();
	reinterpret_cast<const std::shared_ptr<svmComponents::CusomKernelGeneration>&>(m_algorithmConfig.m_populationGeneration)->setCandGamma(gs.getC(), gs.getGammas());
	*/
	try
	{
		auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);
		m_trainingSvmClassifierElement.launch(population, *m_trainingSet);
		m_population = m_validationElement.launch(population, *m_validationSet);
		auto testPopulation = m_validationTestDataElement.launch(population, *m_testSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
			//auto image = m_createVisualizationElement.launch(population, *m_trainingSet, *m_validationSet);

			auto svm = population.getBestOne().getClassifier();
			svmComponents::SvmVisualization visualization;
			auto image = visualization.createDetailedVisualization(*svm,
			                                                       m_algorithmConfig.m_svmConfig.m_height,
			                                                       m_algorithmConfig.m_svmConfig.m_width,
			                                                       *m_trainingSet, *m_validationSet, *m_testSet);

			m_savePngElement.launch(image, m_pngNameSource);
		}

		regionBasedScores();

		logResults(m_population, testPopulation);
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		//m_logger.LOG(logger::LogLevel::Error, exception.what());
	}
}

void CustomKernelWorkflow::regionBasedScores()
{
	auto tr = std::filesystem::path(R"(D:\PHD\experiments\2D_custom_gamma_check\1\train.csv)");
	auto r1 = std::filesystem::path(R"(D:\PHD\experiments\2D_custom_gamma_check\1\region1.csv)");
	auto r2 = std::filesystem::path(R"(D:\PHD\experiments\2D_custom_gamma_check\1\region2.csv)");

	LocalFileDatasetLoader fl(tr, r1, r2);

	/*strategies::TabularDataProviderStrategy tdps;
	static auto region1Data = tdps.launch(R"(D:\PHD\experiments\2D_custom_gamma_check\1\region1.csv)");
	static auto region2Data = tdps.launch(R"(D:\PHD\experiments\2D_custom_gamma_check\1\region2.csv)");*/
	auto region1Data = fl.getValidationSet();
	auto region2Data = fl.getTestSet();

	auto copy = m_population;
	auto region1 = m_validationTestDataElement.launch(copy, region1Data); //region1

	auto copy2 = m_population;
	auto region2 = m_validationTestDataElement.launch(copy2, region2Data); //region2

	auto bestOneConfustionMatrix = region1.getBestOne().getConfusionMatrix().value();
	m_region1Logger.createLogEntry(region1, region1, m_timer, "Region_1", m_generationNumber, svmComponents::Accuracy(bestOneConfustionMatrix), 0);

	auto bestOneConfustionMatrix2 = region2.getBestOne().getConfusionMatrix().value();
	m_region2Logger.createLogEntry(region1, region1, m_timer, "Region_2", m_generationNumber, svmComponents::Accuracy(bestOneConfustionMatrix2), 0);
}
} // namespace genetic
