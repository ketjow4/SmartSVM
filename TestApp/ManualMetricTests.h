#pragma once

// #include "Eigen/Eigen"
// #include "Eigen/Eigenvalues"

#include "Commons.h"
#include "SvmLib/EnsembleSvm.h"
#include "libSvmComponents/DataNormalization.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "libSvmComponents/SvmTraining.h"
#include "libSvmComponents/SvmKernelTraining.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "libSvmComponents/CustomKernelTraining.h"

#include "SvmLib/libSvmInternal.h"

inline svmComponents::SvmTrainingSetChromosome createIndividual(gsl::span<const float>& targets, std::vector<int64_t>& manualDataset)
{
	std::vector<svmComponents::DatasetVector> chromosomeDataset;
	for (auto i = 0; i < manualDataset.size(); ++i)
	{
		chromosomeDataset.emplace_back(svmComponents::DatasetVector(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]])));
	}

	auto manual = svmComponents::SvmTrainingSetChromosome(std::move(chromosomeDataset));
	return manual;
}

inline dataset::Dataset<std::vector<float>, float> createDatasetFromIds(const dataset::Dataset<std::vector<float>, float>& dataset, std::vector<uint64_t>& manualDataset)
{
	auto targets = dataset.getLabels();
	std::vector<svmComponents::DatasetVector> chromosomeDataset;
	for (auto i = 0; i < manualDataset.size(); ++i)
	{
		chromosomeDataset.emplace_back(svmComponents::DatasetVector(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]])));
	}

	auto manual = svmComponents::SvmTrainingSetChromosome(std::move(chromosomeDataset));
	auto shrinkedSet = manual.convertChromosome(dataset);
	return shrinkedSet;
}

inline void test_metric()
{
	auto dataFolder = R"(C:\PHD\experiments_2D4\metric_test\1)";
	//auto dataFolder = R"(C:\PHD\experiments_2D4\linear_easy\1)";
	//auto dataFolder = R"(C:\PHD\experiments_2D4\A1_500\1)";
	//auto dataFolder = R"(C:\PHD\experiments_2D4\A1_501\1)";
	//auto dataFolder = R"(C:\PHD\experiments_2D4\test_1\1)";


	auto linear = genetic::DefaultMemeticConfig::getDefault();
	auto config = linear;

	testApp::ConfigManager configManager;
	configManager.setupDataset(config, dataFolder, "MetricTest", dataFolder);
	configManager.setRandomNumberGenerators(config);
	configManager.saveConfigToFileFolds(config, dataFolder, "MetricTest", 1);


	config.putValue<std::string>("Svm.Metric", "HyperplaneDistance");
	svmComponents::DataNormalization::useDefinedMinMax(0, 500);



	auto outputfolderName = testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath") + "_metric");
	outputfolderName.push_back('\\');
	config.putValue<std::string>("Svm.OutputFolderPath", outputfolderName);


	// m_constKernel(phd::svm::KernelTypes::Linear, { 1 }, false)
	svmComponents::SvmTraining<svmComponents::SvmTrainingSetChromosome> training{
		svmComponents::SvmAlgorithmConfiguration{config},
		svmComponents::SvmKernelChromosome{phd::svm::KernelTypes::Rbf, {120.195213, 1059.671045}, false}, // {C ,gamma} in vector
		//svmComponents::SvmKernelChromosome{phd::svm::KernelTypes::Linear, { 1 }, false}, // {C ,gamma} in vector
		false
	};

	const auto con = genetic::SvmWokrflowConfiguration(config);
	genetic::LocalFileDatasetLoader loader(con.trainingDataPath, con.validationDataPath, con.testDataPath);


	genetic::Timer timer;
	auto targets = loader.getTraningSet().getLabels();

	std::vector<int64_t> manualDataset{ 100, 500, 1000, 1500 ,2000, 2500, 3000, 3500, 4000, 4500, 5000};
	std::vector<int64_t> good_one{ 4705, 3340, 3477, 3816, 847};
	std::vector<int64_t> better_one{ 4705, 3340, 3477, 3816, 847, 2992};
	std::vector<int64_t> good_one_with_single_bad_example{ 4705, 3340, 3477, 3816, 847, 3606 };
	std::vector<int64_t> better_one_with_opposite_close{ 4705, 3340, 3477, 3816, 847, 2992, 39, 3500 };

	//std::vector < int64_t> possible_error{ 2506, 3317, 3771, 3119, 3847, 956, 253, 1620, 3756, 573, 4467, 3123, 2226, 4929, 4190, 2372, }; //this issues is solved
	//std::vector < int64_t> possible_error{ 3637, 5050, 1669, 223, 3001, 3443, 3752, 1136, 1189, 980, 3742, 1755, 2051, 2113, 3848, 4623 };

	//std::vector < int64_t> possible_error{ 1869, 2848, 4092, 1775, 956, 149, 3756, 544, 3713, 1786, 3367, 980, 3742, 965, 3583, 3775 };
	//std::vector < int64_t> possible_error{ 1860,2506,2129,2313,2998,1794,563,78,1266,2214,2228,2216,2661,2516,1559,3272,777,2550,3519,3033,1433,1066,966,1034,82,33,2160,1806, }; //linear_easy problematic case
	//std::vector < int64_t> possible_error{ 337,3477,2680,3676,1042,1066,1685,1219,3340,3990,1496,5970,1353,2117,1739,6212, }; //A1_500 problematic case error in visualization
	//std::vector < int64_t> possible_error{ 100,174,70,6,191,63,235,217,137,61,230,18,178,143,195,122, }; //A1_501 problematic case error with threshold
	//std::vector < int64_t> possible_error{ 7831,12333,8638,15788,5973,8529,7415,6869,7062,15777,4206,10456,12866,13375,6059,11159 }; //test_1 problematic case error with threshold


	std::vector < int64_t> possible_error{ 1986,3151,302,2069,662,815,3198,3542,2281,3032,184,3253,4677,4623,1854,4654, }; //metric_test problematic case error with threshold


	
	//std::vector<svmComponents::SvmTrainingSetChromosome> chromosome_vector;
	
	//auto manual = createIndividual(targets, manualDataset);
	//auto good = createIndividual(targets, good_one);
	//auto better = createIndividual(targets, better_one);
	//auto good_with_bad = createIndividual(targets, good_one_with_single_bad_example);
	//auto better_one_with_opposite = createIndividual(targets, better_one_with_opposite_close);

	auto error = createIndividual(targets, possible_error);


	//geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> pop{ {manual, good, better, good_with_bad, better_one_with_opposite} };
	geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> pop{ {error} };
	training.trainPopulation(pop, loader.getTraningSet());
	
	svmComponents::MemeticTrainingSetEvolutionConfiguration algorithmConfig{ config, loader.getTraningSet() };
	svmStrategies::SvmValidationStrategy<svmComponents::SvmTrainingSetChromosome> validation{ *algorithmConfig.m_svmConfig.m_estimationMethod, false };

	pop = validation.launch(pop, loader.getValidationSet());
	//auto test_pop = pop;
	//test_pop = validation.launch(test_pop, loader.getTestSet());


	std::ofstream out(outputfolderName + "//" + timeUtils::getTimestamp() + "__logs_all.txt");
	for (auto& p : pop)
	{
		out << "Fitness: " << p.getFitness() << std::endl;
	}
	out.close();
	
	int h = 0;
	for (auto& p : pop)
	{
		//svmComponents::MemeticTrainingSetEvolutionConfiguration algorithmConfig{ config, loader.getTraningSet() };

		
		auto detailsPath = outputfolderName + "\\details\\";
		std::filesystem::create_directories(detailsPath);
		auto out =std::filesystem::path(detailsPath +
			"id__" + std::to_string(h) + "__fitness__" + std::to_string(p.getFitness()) + ".png");

		auto svm = p.getClassifier();

		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());


		svmComponents::SvmVisualization visualization;
		auto image = visualization.createDetailedVisualization(*res,
			algorithmConfig.m_svmConfig.m_height,
			algorithmConfig.m_svmConfig.m_width,
			loader.getTraningSet(), loader.getValidationSet());

		strategies::FileSinkStrategy savePngElement;

		savePngElement.launch(image, out);
		++h;






		std::filesystem::path m_pngNameSource;
		svmComponents::SvmVisualization visualization3;
		strategies::FileSinkStrategy m_savePngElement;

		auto Ids = res->getCertaintyRegion(loader.getTraningSet());
		auto finalResult = createDatasetFromIds(loader.getTraningSet(), Ids);

		auto new_set = visualization3.createVisualizationNewValidationSet(500, 500, finalResult);
		genetic::SvmWokrflowConfiguration config_copy4{ "", "", "", outputfolderName + "\\details\\", std::string("shrinked_set") + "__fitness__" + std::to_string(p.getFitness()), "" };
		setVisualizationFilenameAndFormat(algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy4, h);
		m_savePngElement.launch(new_set, m_pngNameSource);

		
	}


	genetic::GeneticWorkflowResultLogger logger;
	auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
	auto validationDataset = loader.getValidationSet();
	auto featureNumber = validationDataset.getSamples()[0].size();

	logger.createLogEntry(pop, pop, timer, "ManualExperiment", 0,
		Accuracy(bestOneConfustionMatrix),
		featureNumber,
		bestOneConfustionMatrix);
	logger.logToFile(outputfolderName + "\\log.txt");
}
















