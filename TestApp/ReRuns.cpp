
#include "ReRuns.h"

#include "AppUtils/PythonFeatureSelection.h"
#include "LastRegionsScores.h"
#include "libGeneticSvm/DatasetLoaderHelper.h"
#include "libGeneticSvm/EnsembleUtils.h"
#include "ManualMetricTests.h"
#include "libGeneticSvm/GridSearchWorkflow.h"
#include "libPlatform/StringUtils.h"
#include "SvmLib/EnsembleListSvm.h"
#include "libSvmComponents/SvmAccuracyMetric.h"
#include "libSvmComponents/SvmAucMetric.h"

std::vector<std::filesystem::path> getAllFeaturesSets(std::filesystem::path folderPath)
{
	std::vector<std::filesystem::path> configFiles;
	for (auto& file : std::filesystem::recursive_directory_iterator(folderPath))
	{
		if (file.path().extension().string() == ".features")
		{
			configFiles.push_back(file.path());
		}
	}
	std::sort(configFiles.begin(), configFiles.end());
	return configFiles;
}

std::vector<std::filesystem::path> getAllSvms(std::filesystem::path folderPath)
{
	std::vector<std::filesystem::path> configFiles;
	for (auto& file : std::filesystem::recursive_directory_iterator(folderPath))
	{
		if (file.path().extension().string() == ".xml")
		{
			configFiles.push_back(file.path());
		}
	}
	std::sort(configFiles.begin(), configFiles.end());
	return configFiles;
}

uint64_t classifyMulticlass(int classes, std::vector<std::shared_ptr<phd::svm::libSvmImplementation>>& classifiers,
                            std::vector<std::map<int, int>>& labelsMapping, const gsl::span<const float>& sample)
{
	std::vector<float> votes(classes);
	for (auto i = 0; i < classifiers.size(); ++i)
	{
		//auto tempClass = classifiers[i]->classifyHyperplaneDistance(sample);
		auto tempClass2 = classifiers[i]->classify(sample);
		votes[labelsMapping[i][tempClass2]]++; // += tempClass;
	}
	/*std::cout << "[";
	for(auto& v : votes)
	{
		std::cout << v << ", ";
	}
	std::cout << "]\n";*/

	auto finalAnswer = std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
	//std::cout << finalAnswer;
	return finalAnswer;
}

void getLabelMapping(std::vector<std::string>::size_type numberOfClasses, std::vector<std::map<int, int>>& labelsMapping)
{
	for (auto i = 0; i < numberOfClasses; ++i)
	{
		for (auto j = i + 1; j < numberOfClasses; ++j)
		{
			std::map<int, int> map;

			map[0] = i;
			map[1] = j;
			labelsMapping.emplace_back(map);
		}
	}
}

void join_multiclass()
{
	auto path = R"(C:\PHD\multiclass_results_more_10_repeats)";

	auto binary_problems = testApp::listDirectories(path);

	auto datasets_multiclass = testApp::listDirectories(R"(C:\PHD\multiclass_binary_join)");

	std::map<std::string, std::vector<std::string>> gropped_datasets;
	for (auto& problem : binary_problems)
	{
		auto key = problem.substr(0, problem.find("__"));

		gropped_datasets[key].emplace_back(problem);
	}

	for (auto& pair : gropped_datasets)
	{
		auto datasetName = std::filesystem::path(pair.first).stem().string();
		//std::filesystem::path(datasets_multiclass[0]).stem().string() == datasetName;
		std::string datasetFolder = *std::find_if(datasets_multiclass.begin(), datasets_multiclass.end(), [&](std::string& value)
		{
			return std::filesystem::path(value).stem().string() == datasetName;
		});

		std::cout << "Running " << pair.first << "\n";
		auto binary_folder_single_dataset = pair.second;

		for (auto al = 0; al < 4; al++)
		{
			std::vector<std::string> algoNames = {"ICPR", "Sequential_gamma_Kernel_Max", "Sequential_gamma_Kernel_Min", "Sequential_gamma_no_alpha"};
			//for each dataset
			std::vector<std::shared_ptr<phd::svm::libSvmImplementation>> classifiers;
			for (auto d : binary_folder_single_dataset)
			{
				auto svms_single = getAllSvms(d);
				auto svm = std::make_shared<phd::svm::libSvmImplementation>(svms_single[0 + al * 10]);
				classifiers.emplace_back(svm);
			}

			std::map<int, int> number_of_classes;
			for (auto i = 2; i < 20; ++i)
			{
				number_of_classes[(i * (i - 1)) / 2] = i;
			}
			auto numberOfClasses = number_of_classes[binary_folder_single_dataset.size()];

			//C:\PHD\multiclass_binary_join
			std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
			auto train = std::filesystem::path(datasetFolder + "\\1\\train.csv");
			auto val = std::filesystem::path(datasetFolder + "\\1\\validation.csv");
			auto test = std::filesystem::path(datasetFolder + "\\1\\test.csv");
			//train02.csv
			ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(train, val, test);

			auto testSet = ptrToLoader->getTestSet();
			std::vector<std::map<int, int>> labelsMapping;
			getLabelMapping(numberOfClasses, labelsMapping);
			std::vector<std::vector<uint32_t>> matrix;
			matrix.resize(numberOfClasses);
			for (auto& row : matrix)
			{
				row.resize(numberOfClasses);
			}
			auto samples = testSet.getSamples();
			auto targets = testSet.getLabels();

			/*auto name = std::filesystem::path("").stem().string();
			auto splited = name.substr(0, name.find("__"));
			std::cout << splited << "\n";*/
			auto algorithm_name = algoNames[al];
			std::ofstream predictions(datasetFolder + "\\" + algorithm_name + "__" + std::to_string(0) + ".csv");

			for (auto i = 0u; i < samples.size(); i++)
			{
				auto result = static_cast<int>(classifyMulticlass(numberOfClasses, classifiers, labelsMapping, samples[i]));
				predictions << result << "," << targets[i] << "\n";
			}

			predictions.close();
		}
	}
}

void createAnswerTargegFiles()
{
	auto path = R"(C:\PHD\multiclass_binary_join)";

	auto datasets = testApp::listDirectories(path);

	for (auto& dataset : datasets)
	{
		auto folds = testApp::listDirectories(dataset);
		for (auto fold : folds)
		{
			auto algorithms = testApp::listDirectories(fold);
			for (auto algorith : algorithms)
			{
				auto svms = getAllSvms(algorith);

				std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
				auto train = std::filesystem::path(fold + "\\train.csv");
				auto val = std::filesystem::path(fold + "\\validation.csv");
				auto test = std::filesystem::path(fold + "\\test.csv");
				ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(train, val, test);
				int i = 0;
				for (auto& svm : svms)
				{
					auto svmClassifier = std::make_shared<phd::svm::libSvmImplementation>(svm);
					auto testSamples = ptrToLoader->getTestSet().getSamples();
					auto testLabels = ptrToLoader->getTestSet().getLabels();

					auto name = std::filesystem::path(algorith).stem().string();
					auto splited = name.substr(0, name.find("__"));
					std::cout << splited << "\n";
					auto algorithm_name = splited;
					std::ofstream predictions(algorith + "\\" + algorithm_name + "__" + std::to_string(i) + ".csv");

					for (auto i = 0u; i < testSamples.size(); i++)
					{
						auto result = svmClassifier->classify(testSamples[i]);
						predictions << result << "," << testLabels[i] << "\n";
					}

					predictions.close();
					i++;
				}
			}
		}
	}
}

void rerun_models()
{
	//auto basePath = R"(D:\implemntation_test)";
	//auto basePath = R"(D:\ENSEMBLE_CLASSIFICATION_442_Feature_saving)";
	//auto basePath = R"(D:\SESVM_TEST_FEATURES_RESULTS)";
	auto basePath = R"(D:\GECCO_Results\2D_results\2D_ENSEMBLE_810_GECCO_final_2D3)";

	auto datasets = testApp::listDirectories(basePath);

	for (auto& dataset : datasets)
	{
		auto folds = testApp::listDirectories(dataset);
		for (auto fold : folds)
		{
			auto algorithms = testApp::listDirectories(fold);
			for (auto algorith : algorithms)
			{
				if (algorith.find("_rerun_") != std::string::npos)
				{
					continue; //if we run this multiple time omit rerun folders, convinience for debugging
				}
				
				std::cout << algorith << "\n";

				auto svms = getAllSvms(algorith + "\\");
				auto configs = testApp::getAllConfigFiles(algorith + "\\");

				auto basename = platform::stringUtils::splitString(algorith, '\\').back();
				auto algorithms_name = platform::stringUtils::splitString(basename, "__").front();
				algorithms_name = algorithms_name + "_rerun_VIS_";
				auto outputFolder = fold + "\\" + algorithms_name;
				outputFolder = testApp::createOutputFolder(outputFolder);

				//in here evaluate model and save results
				auto config = platform::Subtree(std::filesystem::path(configs[0]));
				const auto con = genetic::SvmWokrflowConfiguration(config);
				std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(
					con.trainingDataPath, con.validationDataPath, con.testDataPath);


				std::shared_ptr<phd::svm::ISvm> svmClassifier;
				
				
				if(algorith.find("EnsembleList_") != std::string::npos)
				{
					//re-run with different classifiers at the end of cascade
					//svmClassifier = std::make_shared<phd::svm::EnsembleListSvm>(svms[0], true, true);
					svmClassifier = std::make_shared<phd::svm::EnsembleListSvm>(svms[0], true,true );
				}
				else
				{
					continue;
					//svmClassifier = std::make_shared<phd::svm::libSvmImplementation>(svms[0]);
				}
				
				//auto svmClassifier = std::make_shared<phd::svm::libSvmImplementation>(svms[0]);

				std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_estimationMethod = svmComponents::SvmMetricFactory::create(
					svmComponents::svmMetricType::Auc);

				svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome> m_valdiationElement(*m_estimationMethod, false);
				svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome> m_valdiationTestDataElement(*m_estimationMethod, true);

				svmComponents::SvmKernelChromosome individual;
				individual.updateClassifier(svmClassifier);
				std::vector<svmComponents::SvmKernelChromosome> popContainer{individual};

				genetic::Population<svmComponents::SvmKernelChromosome> population(popContainer);

				auto m_population = m_valdiationElement.launch(population, ptrToLoader->getValidationSet());
				auto copy = population;
				auto testPopulation = m_valdiationTestDataElement.launch(copy, ptrToLoader->getTestSet());
				genetic::GeneticWorkflowResultLogger m_resultLogger;

				auto trainingSetSize = ptrToLoader->getTraningSet().size();
				auto bestOne = m_population.getBestOne();
				auto bestOneConfustionMatrix = bestOne.getConfusionMatrix().value();
				auto featureNumber = ptrToLoader->getValidationSet().getSamples()[0].size();
				auto bestOneIndex = m_population.getBestIndividualIndex();

				std::shared_ptr<genetic::Timer> m_timer = std::make_shared<genetic::Timer>();

				m_resultLogger.createLogEntry(m_population,
				                              testPopulation,
				                              *m_timer,
											  algorithms_name,
				                              0,
				                              svmComponents::Accuracy(bestOneConfustionMatrix),
				                              featureNumber,
				                              trainingSetSize,
				                              bestOneConfustionMatrix,
				                              testPopulation[bestOneIndex].getConfusionMatrix().value());
				m_resultLogger.clearLog(); //clear header row
				m_resultLogger.createLogEntry(m_population,
					testPopulation,
					*m_timer,
					algorithms_name,
					0,
					svmComponents::Accuracy(bestOneConfustionMatrix),
					featureNumber,
					trainingSetSize,
					bestOneConfustionMatrix,
					testPopulation[bestOneIndex].getConfusionMatrix().value());

				m_resultLogger.logToFile(std::filesystem::path(outputFolder + "\\" + timeUtils::getTimestamp() +  algorithms_name + ".json_summary.txt"));
			}
		}
	}
}


enum setType
{
	train,
	val,
	test
};

//TODO fix LastNodeSchemeAndNode
void divide_dataset_last_node(genetic::LocalFileDatasetLoader& ptrToLoader, phd::svm::EnsembleListSvm* svm_ensemble,
	std::vector<dataset::Dataset<std::vector<float>, float>>& certain, 
	std::vector<dataset::Dataset<std::vector<float>, float>>& uncertain, setType type)
{
	dataset::Dataset<std::vector<float>, float> tr;

	switch (type)
	{
	case train:
		tr = ptrToLoader.getTraningSet();
		break;
	case val:
		tr = ptrToLoader.getValidationSet();
		break;
	case test:
		tr = ptrToLoader.getTestSet();
		break;
	default: throw std::exception("wrong dataset type");
	}
	
	
	certain.emplace_back(dataset::Dataset<std::vector<float>, float>());
	uncertain.emplace_back(dataset::Dataset<std::vector<float>, float>());

	/*for (auto i = 0u; i < tr.size(); ++i)
	{
		auto response = svm_ensemble->LastNodeSchemeAndNode(tr.getSample(i)).first;
		if (response != -100)
		{
			certain[0].addSample(tr.getSample(i), tr.getLabel(i));
		}
		else
		{
			uncertain[0].addSample(tr.getSample(i), tr.getLabel(i));
		}
	}*/

	#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(tr.size()); i++)
	{
		auto response = svm_ensemble->LastNodeSchemeAndNode(tr.getSample(i)).first;
		if (response != -100)
		{
			#pragma omp critical
			certain[0].addSample(tr.getSample(i), tr.getLabel(i));
		}
		else
		{
			#pragma omp critical
			uncertain[0].addSample(tr.getSample(i), tr.getLabel(i));
		}
	}
}


//Reruns EnsembleList (single list) when no regions were saved
void rerun_regions()
{
	auto basePath = R"(D:\GECCO_Results\benchmarks\join_rerun2)";

	auto datasets = testApp::listDirectories(basePath);

	for (auto& dataset : datasets)
	{
		auto folds = testApp::listDirectories(dataset);
		for (auto fold : folds)
		{
			auto algorithms = testApp::listDirectories(fold);
			for (auto algorith : algorithms)
			{
				if (algorith.find("_rerun_") != std::string::npos)
				{
					continue; //if we run this multiple time omit rerun folders, convinience for debugging
				}

				std::cout << algorith << "\n";

				auto svms = getAllSvms(algorith + "\\");
				auto configs = testApp::getAllConfigFiles(algorith + "\\");

				auto basename = platform::stringUtils::splitString(algorith, '\\').back();
				auto algorithms_name = platform::stringUtils::splitString(basename, "__").front();
				auto algorithmName = algorithms_name;
				//auto outputFolder = fold + "\\" + algorithms_name;
				//outputFolder = testApp::createOutputFolder(outputFolder);

				//in here evaluate model and save results
				auto config = platform::Subtree(std::filesystem::path(configs[0]));
				const auto con = genetic::SvmWokrflowConfiguration(config);
				std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(
					con.trainingDataPath, con.validationDataPath, con.testDataPath);


				std::shared_ptr<phd::svm::ISvm> svm;


				for(int svmIndex = 0; svmIndex < svms.size(); svmIndex++)
				{
				
				if (algorith.find("EnsembleList_") != std::string::npos)
				{
					//re-run with different classifiers at the end of cascade
					//svmClassifier = std::make_shared<phd::svm::EnsembleListSvm>(svms[0], true, true);
					svm = std::make_shared<phd::svm::EnsembleListSvm>(svms[svmIndex], true);
				}
				else
				{
					continue;
				}


				auto algorithmPath = fold;
				std::ofstream output(algorithmPath + "\\" + algorithmName + "_LastNode.txt", std::ios_base::app);


				auto avg_sv = 0.0;

				auto svm_ensemble = dynamic_cast<phd::svm::EnsembleListSvm*>(svm.get());
				if (svm_ensemble)
				{
					/*	auto nodesSV = svm_ensemble->getNodesNumberOfSupportVectors();

						avg_sv = test_percent[0] * nodesSV[0];

						for(auto i = 1u; i < trainCertain.size(); ++i)
						{
							avg_sv += (test_percent[i] - test_percent[i - 1]) * nodesSV[i];
						}
						avg_sv += (test_percent[test_percent.size() - 1]) * svm_ensemble->getNumberOfSupportVectors();*/

				}
				else
				{
					avg_sv = svm->getNumberOfSupportVectors();
				}



				//create all certain uncertain sets dynamically
				std::vector<dataset::Dataset<std::vector<float>, float>> trainCertain;
				std::vector<dataset::Dataset<std::vector<float>, float>> uncertainTrain;
				std::vector<dataset::Dataset<std::vector<float>, float>> certainVAL;
				std::vector<dataset::Dataset<std::vector<float>, float>> uncertainVAL;
				std::vector<dataset::Dataset<std::vector<float>, float>> certainTEST;
				std::vector<dataset::Dataset<std::vector<float>, float>> uncertainTEST;


				divide_dataset_last_node(*ptrToLoader, svm_ensemble, trainCertain, uncertainTrain, setType::train);
				divide_dataset_last_node(*ptrToLoader, svm_ensemble, certainVAL, uncertainVAL, setType::val);
				divide_dataset_last_node(*ptrToLoader, svm_ensemble, certainTEST, uncertainTEST, setType::test);
				

				for (auto i = trainCertain.size() - 1; i < trainCertain.size(); ++i)
				{
					auto certainLoader = genetic::DatasetLoaderHelper(trainCertain[i], certainVAL[i], certainTEST[i]);
					auto uncertainLoader = genetic::DatasetLoaderHelper(uncertainTrain[i], uncertainVAL[i], uncertainTEST[i]);

					svmComponents::SvmAccuracyMetric acc;
					svmComponents::BaseSvmChromosome individual;
					individual.updateClassifier(svm);
					

					//output << resultTest.m_confusionMatrix << "\t";
					handleCertain(output, certainLoader.getTestSet(), individual);
					handleUncertain(output, uncertainLoader.getTestSet(), individual);

					//output << resultVal.m_confusionMatrix << "\t";
					handleCertain(output, certainLoader.getValidationSet(), individual);
					handleUncertain(output, uncertainLoader.getValidationSet(), individual);

					//output << resultTr.m_confusionMatrix << "\t";
					handleCertain(output, certainLoader.getTraningSet(), individual);
					handleUncertain(output, uncertainLoader.getTraningSet(), individual);

					output << svm->getNumberOfSupportVectors() << "\t";

					auto svm_ensemble = dynamic_cast<phd::svm::EnsembleListSvm*>(svm.get());
					if (svm_ensemble)
					{
						output << svm_ensemble->list_length << "\t";
					}
					else
					{
						output << 0 << "\t";
					}

					output << avg_sv << "\n";
				}

				} //For all svms
			}
		}
	}
}





dataset::Dataset<std::vector<float>, float> getCertain2(std::shared_ptr<phd::svm::ISvm> svm,
	const dataset::Dataset<std::vector<float>, float>& set, bool certain)
{
	std::vector<uint64_t> vset;
	for (auto i = 0u; i < set.size(); ++i)
	{
		vset.emplace_back(i);
	}

	if (certain)
	{
		auto [vectors, ids] = genetic::getCertainDataset(set, vset, svm);
		svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(vectors) };
		auto validationCertainDataset = uncertainTraining.convertChromosome(set);

		return validationCertainDataset;
	}
	else
	{
		auto [vectors, ids] = genetic::getUncertainDataset(set, vset, svm);
		svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(vectors) };
		auto validationCertainDataset = uncertainTraining.convertChromosome(set);

		return validationCertainDataset;
	}
}


std::string scoreEnsemble(std::shared_ptr<phd::svm::EnsembleListSvm> ensemble, genetic::LocalFileDatasetLoader* m_loadingWorkflow)
{

	auto m_metric = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc);
	//svmStrategies::SvmValidationStrategy<svmComponents::BaseSvmChromosome> m_validation(*m_metric, false);
	svmStrategies::SvmValidationStrategy<svmComponents::BaseSvmChromosome> m_validationTest(*m_metric, true);
	
	svmComponents::BaseSvmChromosome to_test2;
	to_test2.updateClassifier(ensemble);
	std::vector<svmComponents::BaseSvmChromosome> vec2;
	vec2.emplace_back(to_test2);

	geneticComponents::Population<svmComponents::BaseSvmChromosome> pop2{ std::move(vec2) };

	m_validationTest.launch(pop2, m_loadingWorkflow->getValidationSet());
	auto validationCM = pop2.getBestOne().getConfusionMatrix();
	auto validationFitness = pop2.getBestOne().getFitness();
	auto SvNumber = pop2.getBestOne().getNumberOfSupportVectors();

	m_validationTest.launch(pop2, m_loadingWorkflow->getTestSet());
	auto testCM = pop2.getBestOne().getConfusionMatrix();
	auto testFitness = pop2.getBestOne().getFitness();

	auto validationCertain = getCertain2(ensemble, m_loadingWorkflow->getValidationSet(), true);

	m_validationTest.launch(pop2, validationCertain);
	auto validationCertainFitness = pop2.getBestOne().getFitness();
	auto validationCertainCM = pop2.getBestOne().getConfusionMatrix();

	auto testCertain = getCertain2(ensemble, m_loadingWorkflow->getTestSet(), true);

	m_validationTest.launch(pop2, testCertain);
	auto testCertainFitness = pop2.getBestOne().getFitness();
	auto testCertainCM = pop2.getBestOne().getConfusionMatrix();

	auto validationUncertain = getCertain2(ensemble, m_loadingWorkflow->getValidationSet(), false);

	m_validationTest.launch(pop2, validationUncertain);
	auto validationUncertainFitness = pop2.getBestOne().getFitness();
	auto validationUncertainCM = pop2.getBestOne().getConfusionMatrix();

	auto testUncertain = getCertain2(ensemble, m_loadingWorkflow->getTestSet(), false);

	m_validationTest.launch(pop2, testUncertain);
	auto testUncertainFitness = pop2.getBestOne().getFitness();
	auto testUncertainCM = pop2.getBestOne().getConfusionMatrix();

	try
	{
		auto sep = "\t";
		///auto path = m_config.outputFolderPath.string() + "//ensembleLog.txt";
		std::stringstream ensembleFile;
		//if (is_empty(path))
		//{
		//	ensembleFile << "### length\tSV \tAuc V\tAuc Test\tConfusion matrix validation(4 numbers)\tConfusion matrix test(4 numbers)\n";
		//}
		ensembleFile << ensemble->list_length << sep << SvNumber << sep << validationFitness << sep << testFitness << sep << validationCM.value() << sep << testCM.value() <<
			sep;
		ensembleFile << validationCertainFitness << sep << validationCertainCM.value() << sep << testCertainFitness << sep << testCertainCM.value() << sep;
		ensembleFile << validationUncertainFitness << sep << validationUncertainCM.value() << sep << testUncertainFitness << sep << testUncertainCM.value() << "\n";

		//ensembleFile.close();

		return ensembleFile.str();
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, exception.what());
		std::cout << exception.what();
		throw;
	}
}

void rerun_regions_for_plots_of_nodes()
{
	//auto basePath = R"(D:\implemntation_test)";
	//auto basePath = R"(D:\ENSEMBLE_CLASSIFICATION_442_Feature_saving)";
	//auto basePath = R"(D:\SESVM_TEST_FEATURES_RESULTS)";
	auto basePath = R"(D:\GECCO_Results\benchmarks\join_rerun2)";

	auto datasets = testApp::listDirectories(basePath);

	for (auto& dataset : datasets)
	{
		auto folds = testApp::listDirectories(dataset);
		for (auto fold : folds)
		{
			auto algorithms = testApp::listDirectories(fold);
			for (auto algorith : algorithms)
			{
				if (algorith.find("_rerun_") != std::string::npos)
				{
					continue; //if we run this multiple time omit rerun folders, convinience for debugging
				}

				std::cout << algorith << "\n";

				auto svms = getAllSvms(algorith + "\\");
				auto configs = testApp::getAllConfigFiles(algorith + "\\");

				auto basename = platform::stringUtils::splitString(algorith, '\\').back();
				auto algorithms_name = platform::stringUtils::splitString(basename, "__").front();
				//algorithms_name = algorithms_name + "_rerun_GS_";
				auto algorithmName = algorithms_name;
				//auto outputFolder = fold + "\\" + algorithms_name;
				//outputFolder = testApp::createOutputFolder(outputFolder);

				//in here evaluate model and save results
				auto config = platform::Subtree(std::filesystem::path(configs[0]));
				const auto con = genetic::SvmWokrflowConfiguration(config);
				std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(
					con.trainingDataPath, con.validationDataPath, con.testDataPath);


				std::shared_ptr<phd::svm::ISvm> svm;


				for (int svmIndex = 0; svmIndex < svms.size(); svmIndex++)
				{

					if (algorith.find("EnsembleList_") != std::string::npos)
					{
						//re-run with different classifiers at the end of cascade
						//svmClassifier = std::make_shared<phd::svm::EnsembleListSvm>(svms[0], true, true);
						svm = std::make_shared<phd::svm::EnsembleListSvm>(svms[svmIndex], true);
					}
					else
					{
						continue;
						//svm = std::make_shared<phd::svm::libSvmImplementation>(svms[0]);
					}


					auto algorithmPath = algorith;
					std::ofstream output(algorithmPath + "\\" + algorithmName + "ensembleLog__" + std::to_string(svmIndex) + "__.txt");
					auto svm_ensemble = dynamic_cast<phd::svm::EnsembleListSvm*>(svm.get());
					


					auto m_newClassificationScheme = true;
					auto list = svm_ensemble->root;

					auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(list->m_svm.get());
					auto svmCopy1 = std::shared_ptr<phd::svm::libSvmImplementation>(res, [=](phd::svm::libSvmImplementation*)
						{
							/*std::cout << "Do nothing on delete to this copy*/
						});
					std::shared_ptr<phd::svm::ListNodeSvm> tempList = std::make_shared<phd::svm::ListNodeSvm>(nullptr, svmCopy1);
					auto tempList2 = tempList;
					list = list->m_next;
					int length = 0;

					std::shared_ptr<phd::svm::ListNodeSvm> listWithoutLastNode = std::make_shared<phd::svm::ListNodeSvm>(nullptr, svmCopy1);
					auto listWithoutLastNodeCreation = listWithoutLastNode;

					auto tempEnsembleList0 = std::make_shared<phd::svm::EnsembleListSvm>(tempList2, length, m_newClassificationScheme);

					auto logs = scoreEnsemble(tempEnsembleList0, ptrToLoader.get());
					output << logs;
					
					while (list)
					{
						auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(list->m_svm.get());
						auto svmCopy = std::shared_ptr<phd::svm::libSvmImplementation>(res2, [=](phd::svm::libSvmImplementation*)
							{
								/*std::cout << "Do nothing on delete to this copy*/
							});
						auto node = std::make_shared<phd::svm::ListNodeSvm>(nullptr, svmCopy);
						tempList->m_next = node;
						tempList = tempList->m_next;
						length++;

						auto tempEnsembleList = std::make_shared<phd::svm::EnsembleListSvm>(tempList2, length, m_newClassificationScheme);
						
						logs = scoreEnsemble(tempEnsembleList, ptrToLoader.get());
						output << logs;


						auto anotherList = std::make_shared<phd::svm::ListNodeSvm>(nullptr, svmCopy);
						listWithoutLastNodeCreation->m_next = anotherList;
						listWithoutLastNodeCreation = listWithoutLastNodeCreation->m_next;

						list = list->m_next;
					}
					

				} //For all svms
			}
		}
	}
}
























double get_margin(std::shared_ptr<phd::svm::ISvm> svm, const dataset::Dataset<std::vector<float>, float>& validation_set)
{
	if (validation_set.empty())
	{
		throw svmComponents::EmptyDatasetException(svmComponents::DatasetType::ValidationOrTest);
	}
	if (!svm->isTrained())
	{
		throw svmComponents::UntrainedSvmClassifierException();
	}

	auto samples = validation_set.getSamples();
	auto targets = validation_set.getLabels();
	double margin = 0.0;

	std::vector<double> probabilites_zeros;
	std::vector<double> probabilites_ones;

	probabilites_zeros.reserve(targets.size());
	probabilites_ones.reserve(targets.size());

	for (auto i = 0u; i < targets.size(); i++)
	{
		if (svm->classifyWithOptimalThreshold(samples[i]) == 0)
		{
			probabilites_zeros.emplace_back(svm->classifyHyperplaneDistance(samples[i]));
		}
		else
		{
			probabilites_ones.emplace_back(svm->classifyHyperplaneDistance(samples[i]));
		}
	}

	std::sort(probabilites_zeros.begin(), probabilites_zeros.end());
	std::sort(probabilites_ones.begin(), probabilites_ones.end());

	margin = std::fabs(probabilites_ones[0] - probabilites_zeros[probabilites_zeros.size() - 1]);

	return margin;
}

// double get_margin_sv(std::shared_ptr<phd::svm::ISvm> svm)
// {
// 	if (!svm->isTrained())
// 	{
// 		throw svmComponents::UntrainedSvmClassifierException();
// 	}
// 	auto supportVectors = svm->getSupportVectors();

// 	std::vector<double> probabilites_zeros;
// 	std::vector<double> probabilites_ones;
// 	probabilites_zeros.reserve(supportVectors.rows);
// 	probabilites_ones.reserve(supportVectors.rows);

// 	auto svmPtr = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
// 	auto svLabels = svmPtr->getSvLables();

// 	for (int i = 0; i < supportVectors.rows; i++)
// 	{
// 		std::vector<float> supportVector;
// 		supportVectors.row(i).copyTo(supportVector);

// 		if (svLabels[i] == 1)
// 		{
// 			probabilites_ones.emplace_back(svm->classifyHyperplaneDistance(supportVector));
// 		}
// 		else
// 		{
// 			probabilites_zeros.emplace_back(svm->classifyHyperplaneDistance(supportVector));
// 		}
// 	}

// 	std::sort(probabilites_zeros.begin(), probabilites_zeros.end());
// 	std::sort(probabilites_ones.begin(), probabilites_ones.end());

// 	double margin = std::fabs(probabilites_ones[0] - probabilites_zeros[probabilites_zeros.size() - 1]);

// 	return margin;
// }

void rerun_models2(std::string basePath_, std::string outputPath, std::string variantName, std::string test_val, std::string metric)
{
	//auto basePath = R"(D:\journal_our_models\Original)";
	//auto basePath = R"(D:\journal_our_models\FSALMA)";
	//auto basePath = R"(D:\journal_our_models\Original_working_models)";
	//auto basePath = R"(D:\journal_our_models\Mixed_kernel_coevolution)";
	//auto basePath = R"(D:\journal_our_models\coevolution\joined)";
	//auto basePath = R"(D:\journal_our_models\journal_datasets_Coevolution_with_PSO_and_others)";

	//auto outputBasePath = R"(D:\rerun_experiment_no_thr_mixed_kernel_coevolution)";
	//auto datasetPath = R"(D:\journal_datasets_rerun_nothr)";
	//auto datasetPath = R"(D:\test_3_data)";
	auto datasetPath = R"(C:\PHD\experiments3)";

	auto basePath = basePath_;
	auto outputBasePath = outputPath;

	auto datasets = testApp::listDirectories(basePath);

	for (auto& dataset : datasets)
	{
		auto folds = testApp::listDirectories(dataset);
		for (auto fold : folds)
		{
			auto algorithms = testApp::listDirectories(fold);
			for (auto algorith : algorithms)
			{
				std::cout << algorith << "\n";

				auto svms = getAllSvms(algorith + "\\");
				auto configs = testApp::getAllConfigFiles(algorith + "\\");

				std::string s = std::filesystem::path(algorith).stem().string();
				std::string delimiter = "__";
				std::string token = s.substr(0, s.find(delimiter));

				auto algorithmName = token + "__" + variantName;

				auto datasetName = *(platform::stringUtils::splitString(fold, '\\').end() - 2) + "\\" + std::filesystem::path(fold).stem().string();
				if ("2D_custom_gamma_check_v2" == platform::stringUtils::splitString(fold, '\\')[3])
					datasetName = "2D_shapes\\" + std::filesystem::path(fold).stem().string();

				auto outputFolder = std::string(outputBasePath) + "\\" + datasetName + "\\" + algorithmName;
				outputFolder = testApp::createOutputFolder(outputFolder);

				//in here evaluate model and save results
				//auto config = platform::Subtree(std::filesystem::path(configs[0]));
				//const auto con = genetic::SvmWokrflowConfiguration(config);
				auto trainingPath = std::filesystem::path(std::string(datasetPath) + "\\" + datasetName + "\\" + "train.csv");
				auto validationPath = std::filesystem::path(std::string(datasetPath) + "\\" + datasetName + "\\" + "validation.csv");
				auto testPath = std::filesystem::path(std::string(datasetPath) + "\\" + datasetName + "\\" + "test.csv");

				std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;

				auto featuresSets = getAllFeaturesSets(algorith + "\\");
				if (!featuresSets.empty())
				{
					//all features sets for fsalma are the same
					std::ifstream featuresFile(featuresSets[0]);

					std::vector<bool> features;
					int temp;
					while (featuresFile >> temp)
					{
						features.emplace_back(static_cast<bool>(temp));
					}

					ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(trainingPath, validationPath, testPath, features);
				}
				else
				{
					ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(trainingPath, validationPath, testPath);
				}

				std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_estimationMethod = std::make_unique<svmComponents::SvmAucMetric>(metric);
				//std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_estimationMethod = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc);

				svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>* m_valdiationElement;
				svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>* m_valdiationTestDataElement;

				if (test_val == "test")
				{
					m_valdiationElement = new svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>(*m_estimationMethod, true);
					m_valdiationTestDataElement = new svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>(*m_estimationMethod, false);
				}
				else if (test_val == "val")
				{
					m_valdiationElement = new svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>(*m_estimationMethod, false);
					m_valdiationTestDataElement = new svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>(*m_estimationMethod, true);
				}

				genetic::GeneticWorkflowResultLogger m_resultLogger;

				const size_t iterationCount = std::distance(svms.begin(), svms.end());
				auto first = svms.begin();

				//#pragma omp parallel for
				for (int i = 0; i < static_cast<int>(iterationCount); i++)
				{
					auto& svm_model_path = *(first + i);
					//std::shared_ptr<phd::svm::libSvmImplementation> svmClassifier;
					std::shared_ptr<phd::svm::EnsembleListSvm> svmClassifier;
#pragma omp critical
					{
						//svmClassifier = std::make_shared<phd::svm::libSvmImplementation>(svm_model_path, ptrToLoader->getTraningSet());

						svmClassifier = std::make_shared<phd::svm::EnsembleListSvm>(svm_model_path);
					}

					//testApp::saveSvmResultsToFile2(outputFolder + "\\", i, *svmClassifier, *ptrToLoader);
					svmComponents::SvmKernelChromosome individual;
					individual.updateClassifier(svmClassifier);
					std::vector<svmComponents::SvmKernelChromosome> popContainer{individual};

					genetic::Population<svmComponents::SvmKernelChromosome> population(popContainer);

					genetic::Population<svmComponents::SvmKernelChromosome> m_population;
					genetic::Population<svmComponents::SvmKernelChromosome> testPopulation;
					if (test_val == "test")
					{
						//use for setting test threshold 
						testPopulation = m_valdiationTestDataElement->launch(population, ptrToLoader->getTestSet());
						auto copy = population;
						m_population = m_valdiationElement->launch(copy, ptrToLoader->getValidationSet());
					}
					else if (test_val == "val")
					{
						m_population = m_valdiationElement->launch(population, ptrToLoader->getValidationSet());
						auto copy = population;
						testPopulation = m_valdiationTestDataElement->launch(copy, ptrToLoader->getTestSet());
					}
					//auto margin = get_margin(svmClassifier, ptrToLoader->getValidationSet());
					//auto sv_margin = get_margin_sv(svmClassifier);

					//testApp::saveSvmResultsToFile3(outputFolder, i, *svmClassifier, *ptrToLoader);

					auto trainingSetSize = ptrToLoader->getTraningSet().size();
					auto bestOne = m_population.getBestOne();
					auto bestOneConfustionMatrix = bestOne.getConfusionMatrix().value();
					auto featureNumber = ptrToLoader->getValidationSet().getSamples()[0].size();
					auto bestOneIndex = m_population.getBestIndividualIndex();

					std::shared_ptr<genetic::Timer> m_timer = std::make_shared<genetic::Timer>();

					//std::cout << "Test matrix: " << testPopulation[bestOneIndex].getConfusionMatrix().value() << "\n";
					//std::cout << "MCC: " << variantName << "  " << testPopulation[bestOneIndex].getConfusionMatrix().value().MCC() << "\n";

#pragma omp critical
					{
						m_resultLogger.createLogEntry(m_population,
						                              testPopulation,
						                              *m_timer,
						                              algorithmName,
						                              0,
						                              svmComponents::Accuracy(bestOneConfustionMatrix),
						                              featureNumber,
						                              trainingSetSize,
						                              bestOneConfustionMatrix,
						                              testPopulation[bestOneIndex].getConfusionMatrix().value());
						//margin,
						//sv_margin
					}
				}

				m_resultLogger.logToFile(std::filesystem::path(outputFolder + "\\" + algorithmName + ".json_summary.txt"));
			}
		}
	}
}
