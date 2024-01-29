// #pragma once

// // #include "Eigen/Eigen"
// // #include "Eigen/Eigenvalues"

// #include "Commons.h"
// #include "SvmLib/EnsembleSvm.h"
// #include "libSvmComponents/DataNormalization.h"
// #include "libSvmComponents/CustomWidthGauss.h"
// #include "libSvmComponents/SvmTraining.h"
// #include "libSvmComponents/SvmKernelTraining.h"
// #include "SvmLib/libSvmImplementation.h"
// #include "libSvmComponents/ConfusionMatrixMetrics.h"
// #include "libSvmComponents/CustomKernelTraining.h"


// struct AucResults
// {
// 	AucResults(svmComponents::ConfusionMatrix matrix, double aucValue, double optimalThreshold)
// 		: m_matrix(std::move(matrix))
// 		, m_aucValue(aucValue)
// 		, m_optimalThreshold(optimalThreshold)
// 	{
// 	}

// 	const svmComponents::ConfusionMatrix m_matrix;
// 	const double m_aucValue;
// 	const double m_optimalThreshold;
// };

// double trapezoidArea(double x1, double x2, double y1, double y2)
// {
// 	auto base = std::fabs(x1 - x2);
// 	auto height = (y1 + y2) / 2.0;
// 	return height * base;
// }

// AucResults auc(std::vector<std::pair<double, int>>& probabilityTargetPair, int negativeCount, int positiveCount)
// {
// 	std::sort(probabilityTargetPair.begin(), probabilityTargetPair.end(), [](const auto& a, const auto& b)
// 		{
// 			return a.first > b.first;
// 		});

// 	double auc = 0;
// 	double previousProbability = -1;
// 	int falsePositive = 0;
// 	int truePositive = 0;
// 	int falsePositivePreviousIteration = 0;
// 	int truePositivesPreviousIteration = 0;

// 	auto maxAccuracyForThreshold = 0.0;
// 	auto threshold = 0.0;
// 	svmComponents::ConfusionMatrix matrixWithOptimalThreshold(0u, 0u, 0u, 0u);

// 	for (const auto& pair : probabilityTargetPair)
// 	{
// 		auto [probability, label] = pair;

// 		if (probability != previousProbability)
// 		{
// 			auc += trapezoidArea(falsePositive, falsePositivePreviousIteration, truePositive, truePositivesPreviousIteration);
// 			previousProbability = probability;
// 			falsePositivePreviousIteration = falsePositive;
// 			truePositivesPreviousIteration = truePositive;
// 		}

// 		label == 1 ? truePositive++ : falsePositive++;

// 		const auto accuracyForThreshold = static_cast<double>(truePositive + (negativeCount - falsePositive)) / static_cast<double>(negativeCount + positiveCount);
// 		if (accuracyForThreshold > maxAccuracyForThreshold)
// 		{
// 			matrixWithOptimalThreshold = svmComponents::ConfusionMatrix(truePositive, (negativeCount - falsePositive),
// 				falsePositive, (positiveCount - truePositive));
// 			maxAccuracyForThreshold = accuracyForThreshold;
// 			threshold = probability;
// 		}

// 	}
// 	auc += trapezoidArea(negativeCount, falsePositivePreviousIteration, positiveCount, truePositivesPreviousIteration);
// 	auc /= (static_cast<double>(positiveCount) * static_cast<double>(negativeCount));

// 	return AucResults(matrixWithOptimalThreshold, auc, threshold);
// }

// svmComponents::Metric ensembleValidation(const dataset::Dataset<std::vector<float>, float>& data,
// 	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome>& population)
// {
// 	auto samples = data.getSamples();
// 	auto targets = data.getLabels();

// 	std::vector<float> hyperplaneDistancePerSample;
// 	hyperplaneDistancePerSample.resize(samples.size());

// 	for (auto& individual : population)
// 	{
// 		auto classifier = individual.getClassifier();
// 		for (auto f = 0; f < samples.size(); ++f)
// 		{
// 			hyperplaneDistancePerSample[f] += classifier->classifyHyperplaneDistance(samples[f]);
// 		}
// 	}

// 	auto positiveCount = static_cast<unsigned int>(std::count_if(targets.begin(), targets.end(),
// 		[](const auto& target)
// 		{
// 			return target == 1;
// 		}));
// 	auto negativeCount = static_cast<unsigned int>(samples.size() - positiveCount);

// 	std::vector<std::pair<double, int>> probabilites;
// 	probabilites.reserve(targets.size());

// 	for (auto i = 0u; i < targets.size(); i++)
// 	{
// 		probabilites.emplace_back(std::make_pair(hyperplaneDistancePerSample[i], static_cast<int>(targets[i])));
// 	}
// 	AucResults result = auc(probabilites, negativeCount, positiveCount);
// 	population[0].getClassifier()->setOptimalProbabilityThreshold(result.m_optimalThreshold);  //TODO think how to use it in ensemble
// 	population[1].getClassifier()->setOptimalProbabilityThreshold(result.m_optimalThreshold);
// 	return svmComponents::Metric(result.m_aucValue, result.m_matrix);
// }

// void buildEnsembleFromLastGeneration(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome>& population,
// 	genetic::LocalFileDatasetLoader& loader)
// {
// 	auto svNumber = 0;
// 	for (auto& p : population)
// 	{
// 		svNumber += p.getNumberOfSupportVectors();
// 	}

// 	auto validationResults = ensembleValidation(loader.getValidationSet(), population);
// 	auto testResults = ensembleValidation(loader.getTestSet(), population);
	

// 	auto validationDataset = loader.getValidationSet();
// 	auto featureNumber = validationDataset.getSamples()[0].size();


// 	std::string logInfo;
// 	logInfo += "Manual ensemble selection";
// 	logInfo += "\t";
// 	logInfo += std::to_string(validationResults.m_fitness).append("\t");
// 	logInfo += std::to_string(validationResults.m_fitness).append("\t");
// 	logInfo += std::to_string(svNumber).append("\t");
// 	logInfo += std::to_string(svNumber).append("\t");
// 	logInfo += std::to_string(testResults.m_fitness).append("\t");
// 	logInfo += std::to_string(testResults.m_fitness).append("\t");
// 	logInfo += std::to_string(svmComponents::Accuracy(validationResults.m_confusionMatrix.value())).append("\t");
// 	logInfo += std::to_string(featureNumber).append("\t");

// 	logInfo += validationResults.m_confusionMatrix.value().to_string().append("\t");
// 	logInfo += testResults.m_confusionMatrix.value().to_string().append("\t");
// 	logInfo.append("\n");

// 	std::cout << logInfo;
// }

// //scalling needed
// //std::vector<int64_t> combined_ids{ 3238, 4857, 2866, 2251, 356, 1697, 1291, 3892, 1267, 3822, 749,  };
// //std::vector<int64_t> combined_gammas{ -1, -1, -1, -1 ,300, 300, 300, 300, 300, 300, 300};

// //other order of vectors -- does not change the end result of optimization 
// //std::vector<int64_t> combined_ids{  356, 1697, 1291, 3892, 1267, 3822, 749, 3238, 4857, 2866, 2251 };
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300, 300, 300, 300, -1, -1, -1, -1 };


// //no scalling needed
// //std::vector<int64_t> combined_ids{ 356, 1697, 1291, 3892, 1750, 933, 1910, 660, 2438, 2112, /*3670,*/ 4857};
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300, 300, 300, 300, 300, -1, -1, /*-1,*/ -1, };


// //no scalling needed - poor example but seems to work with/without scalling, different C values provides the same resutls, different C per kernel does not change the result
// //std::vector<int64_t> combined_ids{ 356, 1697, 1291, 3892, 1750, 933, 1910, 660, 2438, 2112, 4857 };
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300, 300, 300, 300, 300, -1, -1, -1, };
// //

// //other not working as expected -- fixed, the example had too many training vector compared to gamma values, scalling gives no issues (works with and without), different C for kernels provide poor resutls
// //std::vector<int64_t> combined_ids{ 3704, 3873, 3649, 4434, 4327, 3339, 3856, 1630, 3212, 4002, 495, 3839};
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300 ,300 ,300, 300, 300, 300, 300, 300 ,-1, -1 };


// //working exmaple - C = 1 strange behaviour with linear, seems that liniear need higher C and no scalling or the same C and upsacling alpha for some reason
// //std::vector<int64_t> combined_ids{ 356, 1697, 1291, 3892, 4291, 91, 495, 3813 };
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300, -1, -1, -1, -1 };



// //problematic one - gammas are equal between kernels linear is dominant
// //std::vector<int64_t> combined_ids{ 4317, 516, 3309, 3520, 3874, 675, 1082, 781, 1005, 2981, 3487, 346, 2541, 3809, 2226, 2253 };
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300, 300, 300, 300, 300, -1, -1, -1, -1, -1, -1, -1, -1 };


// //std::vector<int64_t> combined_ids{ 1413, 524, 4155, 3246, 4062, 4639, 1926, 616, 4384, 4021, 138, 3760, 3109, 2904, 2446, 2195 };
// //std::vector<int64_t> combined_gammas{ 300, 300, 300, 300, 300, 300, 300, 300, -1, -1, -1, -1, -1, -1, -1, -1 };

// std::vector<int64_t> combined_ids{ 4384, 4021, 138, 3760, 3109, 2904, 2446, 2195, 4155, 524 };
// std::vector<int64_t> combined_gammas{ -1, -1, -1, -1, -1, -1, -1, -1, 50, 50 };

// //std::vector<int64_t> combined_ids{ /*4384,*/ 4021, 138, 2446, 2195 };
// //std::vector<int64_t> combined_gammas{ /*-1,*/ -1, -1, -1, -1 };


// #include "svm/libSvm/libSvmInternal.h"

// void analize_kernel_matrix(const dataset::Dataset<std::vector<float>, float>& trainingSet)
// {
// 	phd::svm::libSvmImplementation svm;

// 	std::vector<svmComponents::Gene> chromosomeDataset;
// 	std::vector<int64_t> manualDataset = combined_ids;
// 	std::vector<int64_t> gammas = combined_gammas;
// 	for (auto i = 0; i < manualDataset.size(); ++i)
// 	{
// 		chromosomeDataset.emplace_back(svmComponents::Gene(manualDataset[i], static_cast<std::uint8_t>(0), gammas[i]));
// 	}
// 	svmComponents::SvmCustomKernelChromosome ch(std::move(chromosomeDataset), 1.0);
// 	auto trainingSetSvm = svm.createDatasetForTraining(ch.convertChromosome(trainingSet));

// 	svm_parameter params = svm.m_param;
// 	std::vector<double> gammas_d{ combined_gammas.begin(), combined_gammas.end() };
// 	params.gammas = &gammas_d;
	
// 	Kernel k(static_cast<int>(combined_gammas.size()), trainingSetSvm.x, params);

// 	Eigen::MatrixXd kernel_matrix;
// 	kernel_matrix.resize(static_cast<Eigen::Index>(combined_gammas.size()), static_cast<Eigen::Index>(combined_gammas.size()));

// 	for(int i = 0; i < static_cast<int>(combined_gammas.size()); i++)
// 	{
// 		for (int j = 0; j < static_cast<int>(combined_gammas.size()); j++)
// 		{
// 			kernel_matrix(i, j) = k.kernerl_rbf_and_linear(i, j);
// 		}
// 	}

// 	std::cout << kernel_matrix << std::endl;

// 	Eigen::EigenSolver< Eigen::MatrixXd> solver(kernel_matrix);
// 	std::cout << "\n";
// 	std::cout << "Eigen values:\n " << solver.eigenvalues() << "\n";
// 	std::cout << "END\n ";
	
	
// }


// std::pair<std::vector<int64_t>, std::vector<int64_t>> get_linear_ids()
// {
// 	std::vector<int64_t> ids;
// 	std::vector<int64_t> gammas;

// 	for(auto i =0u; i < combined_ids.size(); ++i)
// 	{
// 		if(combined_gammas[i] == -1)
// 		{
// 			ids.emplace_back(combined_ids[i]);
// 			gammas.emplace_back(combined_gammas[i]);
// 		}
// 	}
// 	return { ids,gammas };
// }

// std::pair<std::vector<int64_t>, std::vector<int64_t>> get_gammaa_ids_and_gammas()
// {
// 	std::vector<int64_t> ids;
// 	std::vector<int64_t> gammas;

// 	for (auto i = 0u; i < combined_ids.size(); ++i)
// 	{
// 		if (combined_gammas[i] != -1)
// 		{
// 			ids.emplace_back(combined_ids[i]);
// 			gammas.emplace_back(combined_gammas[i]);
// 		}
// 	}
// 	return { ids,gammas };
// }



// inline void manual_setting_rbf_linear(double cValue = 1.0)
// {
// 	auto dataFolder = R"(C:\PHD\experiments\linear_with_pool\1)";

	
// 	auto linear = genetic::DefaultSequentialGammaConfig::getDefault();
// 	linear.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
// 	linear.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");

// 	auto config = linear;
	
// 	testApp::ConfigManager configManager;
// 	configManager.setupDataset(config, dataFolder, "RBF_LINEAR", dataFolder);
// 	configManager.setRandomNumberGenerators(config);
// 	configManager.saveConfigToFileFolds(config, dataFolder, "RBF_LINEAR", 1);

// 	config.putValue<std::string>("Svm.Metric", "AUC");
// 	svmComponents::DataNormalization::useDefinedMinMax(0, 500);


// 	auto outputfolderName = testApp::createOutputFolderWithDetails(config.getValue<std::string>("Svm.OutputFolderPath"),  "KERNEL_TEST_single_SVM_C=" + std::to_string(cValue));
// 	outputfolderName.push_back('\\');
// 	config.putValue<std::string>("Svm.OutputFolderPath", outputfolderName);

// 	svmComponents::SvmTrainingCustomKernel training{ svmComponents::SvmAlgorithmConfiguration{config}, false, "RBF_LINEAR", true };

// 	const auto con = genetic::SvmWokrflowConfiguration(config);
// 	genetic::LocalFileDatasetLoader loader(con.trainingDataPath, con.validationDataPath, con.testDataPath);

// 	analize_kernel_matrix(loader.getTraningSet());

// 	svmComponents::SvmCustomKernelChromosome manual;

// 	std::vector<svmComponents::Gene> chromosomeDataset;
// 	auto smallWidth = 100;
// 	auto large = 10;
// 	auto line = -1;
// 	//std::vector<int64_t> manualDataset{ 2328, 2765,  1009, 1427, /*3529, 4687,*/     4112, 3644,   882, 1544};
// 	//std::vector<int64_t> gammas{ line, line, line, line,  /*line,line,*/                smallWidth,large,large,smallWidth };


// 	/*std::vector<int64_t> manualDataset{ 356, 1697, 1291, 3892, 4291, 91, 495, 3813 };
// 	std::vector<int64_t> gammas{ 300, 300, 300, 300, -1, -1, -1, -1 };*/
// 	std::vector<int64_t> manualDataset = combined_ids;
// 	std::vector<int64_t> gammas = combined_gammas;


// 	std::mt19937 eng1(1);
// 	auto eng2 = eng1;

// 	std::shuffle(begin(manualDataset), end(manualDataset), eng1);
// 	std::shuffle(begin(gammas), end(gammas), eng2);

// 	auto targets = loader.getTraningSet().getLabels();

// 	for (auto i = 0; i < manualDataset.size(); ++i)
// 	{
// 		chromosomeDataset.emplace_back(svmComponents::Gene(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]]), gammas[i]));
// 	}


// 	genetic::Timer timer;

// 	manual = svmComponents::SvmCustomKernelChromosome(std::move(chromosomeDataset), cValue);

// 	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop{ {manual} };

// 	training.trainPopulation(pop, loader.getTraningSet());

// 	//auto svm2 = pop.getBestOne().getClassifier();
// 	//auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm2.get());
// 	/*std::cout << "Alphas: \n";
// 	for (auto i = 0; i < res2->m_model->l; ++i)
// 	{
// 		std::cout << res2->m_model->sv_coef[0][i] << "\n";
// 	}*/

// 	//*res2->m_model->sv_coef = alphas.data();

// 	svmComponents::SequentialGammaConfig algorithmConfig{ config, loader.getTraningSet() };
// 	svmStrategies::SvmValidationStrategy<svmComponents::SvmCustomKernelChromosome> validation{ *algorithmConfig.m_svmConfig.m_estimationMethod, false };



	
// 	pop = validation.launch(pop, loader.getValidationSet());
// 	auto test_pop = pop;
// 	test_pop = validation.launch(test_pop, loader.getTestSet());

// 	filesystem::Path m_pngNameSource;
// 	genetic::setVisualizationFilenameAndFormat(algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, con, 0);

// 	auto svm = pop.getBestOne().getClassifier();

// 	auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
// 	auto [sv_to_vec_map, scores] = res->check_sv(loader.getTraningSet());


// 	//std::vector<float> alphas_changed{ 1,	10,	8.665647, -4.34147, -10, -4.32418, -1 };
// 	//std::vector<float> alphas_changed{ -1.77963,	10,	4.665647, -2.34147, -8.33734, -2.32418,	0.116965 };
// 	//std::vector<float> alphas_changed{ -1.77963,	10,	6, - 3.0, - 8.33734, - 3.0,	0.116965}; //alphy z oddzielnego treningu
// 	//std::vector<float> alphas_changed{ 4.77298,	10,	0.281667, - 2.34147, - 10, - 2.32418, - 0.388993}; //wersja podstawowa
// 	//std::vector<float> alphas_changed{ -1.77963,	3,	10, - 5, - 1.33734, - 5,	0.116965}; //dominacja gammy


// 	/*std::vector<float> alphas_changed{ 1,	1,	1, - 0.33, - 0.33, - 0.33, - 2
// 	};
// 	for (auto i = 0; i < res->m_model->l; ++i)
// 	{
// 		res->m_model->sv_coef[0][i] = alphas_changed[i];
// 	}
// 	pop = validation.launch(pop, loader.getValidationSet());*/
	

// 	//res->m_model->param.gammas_after_training->emplace_back(-1);
// 	//res->m_model->param.gammas_after_training->emplace_back(-1);
// 	//res->m_model->param.gammas_after_training->emplace_back(-1);

// 	//realloc(res->m_model->sv_coef[0], res->m_model->l + 3);
// 	//realloc(res->m_model->SV, res->m_model->l + 3);
	

// 	svmComponents::SvmVisualization visualization;
// 	//visualization.setMap(sv_to_vec_map);
// 	//visualization.setScores(scores);
// 	visualization.setGene(manual);

// 	std::set<double> s{ combined_gammas.begin(), combined_gammas.end() };
// 	std::vector<double> gammas_d{ s.begin(), s.end() };
// 	visualization.setGammasValues(gammas_d);
// 	auto image = visualization.createDetailedVisualization(*res,
// 		algorithmConfig.m_svmConfig.m_height,
// 		algorithmConfig.m_svmConfig.m_width,
// 		loader.getTraningSet(), loader.getValidationSet());

// 	strategies::FileSinkStrategy savePngElement;

// 	savePngElement.launch(image, m_pngNameSource);

// 	std::ofstream out(m_pngNameSource.string() + timeUtils::getTimestamp() + ".txt");
// 	out << "AUC: " << pop.getBestOne().getFitness() << std::endl;
// 	out << "Tr: ";
// 	for (auto v : manual.getDataset())
// 		out << v.id << " ";
// 	out << std::endl;

// 	out << "Gammas: ";
// 	for (auto i = 0; i < res->m_model->l; ++i)
// 		out << res->m_model->param.gammas_after_training->at(i) << " ";
// 	out << std::endl;

// 	out << "Alphas: ";
// 	for (auto i = 0; i < res->m_model->l; ++i)
// 	{
// 		out << res->m_model->sv_coef[0][i] << " ";
// 	}
// 	out << "\nRho: " << res->m_model->rho[0] << "\n";
// 	out << "Thr: " << res->m_model->param.m_optimalProbabilityThreshold << "\n";
// 	out << std::endl;

// 	out.close();
// 	std::cout << pop.getBestOne().getFitness();

// 	genetic::GeneticWorkflowResultLogger logger;
// 	auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
// 	auto validationDataset = loader.getValidationSet();
// 	auto featureNumber = validationDataset.getSamples()[0].size();

// 	logger.createLogEntry(pop, test_pop, timer, "ManualExperiment", 0,
// 		Accuracy(bestOneConfustionMatrix),
// 		featureNumber,
// 		bestOneConfustionMatrix);
// 	logger.logToFile(m_pngNameSource.string() + "log.txt");
// }




// inline void manual_ensemble_rbf_linear()
// {
// 	auto dataFolder = R"(C:\PHD\experiments\linear_with_pool\1)";


// 	auto linear = genetic::DefaultSequentialGammaConfig::getDefault();
// 	linear.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MIN");
// 	//linear.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");

// 	auto config = linear;

// 	testApp::ConfigManager configManager;
// 	configManager.setupDataset(config, dataFolder, "RBF_Ensemble", dataFolder);
// 	configManager.setRandomNumberGenerators(config);
// 	configManager.saveConfigToFileFolds(config, dataFolder, "RBF_Ensemble", 1);


// 	config.putValue<std::string>("Svm.Metric", "AUC");
// 	svmComponents::DataNormalization::useDefinedMinMax(0, 500);

// 	//changeConfigurationForExperiment({ R"(D:\PHD\experiments\2D_custom_gamma_check\)" }, "CustomKernel");

// 	auto outputfolderName = testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath") + "_ensemble");
// 	outputfolderName.push_back('\\');
// 	config.putValue<std::string>("Svm.OutputFolderPath", outputfolderName);

// 	svmComponents::SvmTrainingCustomKernel training{ svmComponents::SvmAlgorithmConfiguration{config}, false, "RBF_MIN", true };

// 	const auto con = genetic::SvmWokrflowConfiguration(config);
// 	genetic::LocalFileDatasetLoader loader(con.trainingDataPath, con.validationDataPath, con.testDataPath);


// 	genetic::Timer timer;
// 	auto targets = loader.getTraningSet().getLabels();

// 	std::vector<svmComponents::Gene> chromosomeDataset;
// 	std::vector<int64_t> manualDatasetrbf = get_gammaa_ids_and_gammas().first;
// 	std::vector<int64_t> gammasrbf = get_gammaa_ids_and_gammas().second;
// 	for (auto i = 0; i < manualDatasetrbf.size(); ++i)
// 	{
// 		chromosomeDataset.emplace_back(svmComponents::Gene(manualDatasetrbf[i], static_cast<std::uint8_t>(targets[manualDatasetrbf[i]]), gammasrbf[i]));
// 	}
// 	svmComponents::SvmCustomKernelChromosome manualrbf = svmComponents::SvmCustomKernelChromosome(std::move(chromosomeDataset), 1.0);


// 	std::vector<int64_t> manualDatasetlin = get_linear_ids().first; // { /*356, 1697, 1291, 3892,*/ 4291, 91, 495, 3813 };
// 	std::vector<int64_t> gammaslin = get_linear_ids().second; // { /*300, 300, 300, 300,*/ -1, -1, -1, -1 };
// 	std::vector<svmComponents::Gene> chromosomeDatasetlin;
// 	for (auto i = 0; i < manualDatasetlin.size(); ++i)
// 	{
// 		chromosomeDatasetlin.emplace_back(svmComponents::Gene(manualDatasetlin[i], static_cast<std::uint8_t>(targets[manualDatasetlin[i]]), gammaslin[i]));
// 	}
// 	svmComponents::SvmCustomKernelChromosome manuallin = svmComponents::SvmCustomKernelChromosome(std::move(chromosomeDatasetlin), 1.0);

// 	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop{ {manualrbf, manuallin} };

// 	training.trainPopulation(pop, loader.getTraningSet());


// 	buildEnsembleFromLastGeneration(pop, loader);

// 	int h = 0;
// 	for (auto& p : pop)
// 	{
// 		svmComponents::SequentialGammaConfig algorithmConfig{ config, loader.getTraningSet() };

// 		filesystem::FileSystem fs;
// 		auto detailsPath = outputfolderName + "\\details\\";
// 		fs.createDirectories(detailsPath);
// 		auto out = filesystem::Path(detailsPath +
// 			"id__" + std::to_string(h) + ".png");

// 		auto svm = p.getClassifier();

// 		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
// 		auto [sv_to_vec_map, scores] = res->check_sv(loader.getTraningSet());

// 		svmComponents::SvmVisualization visualization;
// 		visualization.setMap(sv_to_vec_map);
// 		visualization.setScores(scores);
// 		if (h == 0)
// 			visualization.setGene(manualrbf);
// 		else
// 			visualization.setGene(manuallin);
// 		auto image = visualization.createDetailedVisualization(*res,
// 			algorithmConfig.m_svmConfig.m_height,
// 			algorithmConfig.m_svmConfig.m_width,
// 			loader.getTraningSet(), loader.getValidationSet());

// 		strategies::FileSinkStrategy savePngElement;

// 		savePngElement.launch(image, out);
// 		++h;

// 		std::ofstream file(outputfolderName + "\\ensemble_" + std::to_string(h) + "__" + timeUtils::getTimestamp() + ".txt");
// 		file << "AUC: " << pop.getBestOne().getFitness() << std::endl;
// 		file << "Tr: ";
// 		for (auto v : p.getDataset())
// 			file << v.id << " ";
// 		file << std::endl;

// 		file << "Gammas: ";
// 		for (auto i = 0; i < res->m_model->l; ++i)
// 			file << res->m_model->param.gammas_after_training->at(i) << " ";
// 		file << std::endl;

// 		file << "Alphas: ";
// 		for (auto i = 0; i < res->m_model->l; ++i)
// 		{
// 			file << res->m_model->sv_coef[0][i] << " ";
// 		}
// 		file << "\nRho: " << res->m_model->rho[0] << "\n";
// 		file << std::endl;

// 		file.close();
// 	}

// 	auto rbf = *reinterpret_cast<phd::svm::libSvmImplementation*>(pop[0].getClassifier().get());
// 	auto lin = *reinterpret_cast<phd::svm::libSvmImplementation*>(pop[1].getClassifier().get());
	
// 	//phd::svm::EnsembleSvm esvm (lin, rbf);
// 	////(-1.15, 0.55)
// 	////for(double thr = -1.15; thr < 0.56; thr += 0.01)
// 	//{
// 	//	//std::cout << thr;
// 	//	esvm.setOptimalProbabilityThreshold(lin.m_optimalProbabilityThreshold);

// 	//	std::vector<svmComponents::Gene> combined;
// 	//	std::vector<int64_t> manualDataset = combined_ids;
// 	//	std::vector<int64_t> gammas = combined_gammas;
// 	//	for (auto i = 0; i < manualDataset.size(); ++i)
// 	//	{
// 	//		combined.emplace_back(svmComponents::Gene(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]]), gammas[i]));
// 	//	}
// 	//	svmComponents::SvmCustomKernelChromosome manual = svmComponents::SvmCustomKernelChromosome(std::move(combined), 10.0);

// 	//	svmComponents::SequentialGammaConfig algorithmConfig{ config, loader.getTraningSet() };
// 	//	filesystem::Path m_pngNameSource;
// 	//	genetic::setVisualizationFilenameAndFormat(algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, con, 0);

// 	//	svmComponents::SvmVisualization visualization;
// 	//	visualization.setGene(manual);
// 	//	std::set<double> s{ combined_gammas.begin(), combined_gammas.end() };
// 	//	std::vector<double> gammas_d{ s.begin(), s.end() };
// 	//	visualization.setGammasValues(gammas_d);
// 	//	auto image = visualization.createDetailedVisualization(esvm,
// 	//		algorithmConfig.m_svmConfig.m_height,
// 	//		algorithmConfig.m_svmConfig.m_width,
// 	//		loader.getTraningSet(), loader.getValidationSet());

// 	//	strategies::FileSinkStrategy savePngElement;

// 	//	savePngElement.launch(image, m_pngNameSource);

// 	//	//std::cout << "Line min/max: " << esvm.minLine << "/" << esvm.maxLine << "   RBF min/max: " << esvm.minRBF << "/" << esvm.maxRBF << "\n";
// 	//}
// 	/*genetic::GeneticWorkflowResultLogger logger;
// 	auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
// 	auto validationDataset = loader.getValidationSet();
// 	auto featureNumber = validationDataset.getSamples()[0].size();

// 	logger.createLogEntry(pop, test_pop, timer, "ManualExperiment", 0,
// 		Accuracy(bestOneConfustionMatrix),
// 		featureNumber,
// 		bestOneConfustionMatrix);
// 	logger.logToFile(m_pngNameSource.string() + "log.txt");*/

// }