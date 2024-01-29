#pragma once

#include <iostream>

#include "AppUtils/AppUtils.h"
#include "libGeneticSvm/LocalFileDatasetLoader.h"
#include "libPlatform/StringUtils.h"
#include "SvmLib/EnsembleListSvm.h"
#include "libSvmComponents/BaseSvmChromosome.h"
#include "libSvmComponents/ConfusionMatrix.h"
#include "libSvmComponents/SvmAccuracyMetric.h"
#include "ReRuns.h"



inline std::vector<std::string> getAllCsvs(std::filesystem::path folderPath)
{
	std::vector<std::string> configFiles;
	for (auto& file : std::filesystem::recursive_directory_iterator(folderPath))
	{
		if (file.path().extension().string() == ".csv")
		{
			configFiles.push_back(file.path().string());
		}
	}
	std::sort(configFiles.begin(), configFiles.end());
	return configFiles;
}

inline std::vector<std::string> filterOut(const std::vector<std::string>& input, std::string searchedFor)
{
	std::vector<std::string> result;
	for (auto& path : input)
	{
		if (path.find(searchedFor) != std::string::npos)
		{
			result.emplace_back(path);
		}
	}
	return result;
}

inline bool compareNat(const std::string& a, const std::string& b)
{
	if (a.empty())
		return true;
	if (b.empty())
		return false;
	if (std::isdigit(a[0]) && !std::isdigit(b[0]))
		return true;
	if (!std::isdigit(a[0]) && std::isdigit(b[0]))
		return false;
	if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
	{
		if (std::toupper(a[0]) == std::toupper(b[0]))
			return compareNat(a.substr(1), b.substr(1));
		return (std::toupper(a[0]) < std::toupper(b[0]));
	}

	// Both strings begin with digit --> parse both numbers
	std::istringstream issa(a);
	std::istringstream issb(b);
	int ia, ib;
	issa >> ia;
	issb >> ib;
	if (ia != ib)
		return ia < ib;

	// Numbers are the same --> remove numbers and recurse
	std::string anew, bnew;
	std::getline(issa, anew);
	std::getline(issb, bnew);
	return (compareNat(anew, bnew));
}

inline void regionsScoreExperiment(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);

	auto input_path = config.outputFolder;

	//auto input_path = R"(D:\ENSEMBLE_CLASSIFICATION_412_LASTNODE_FULL_SVM_2D_REGIONS)";

	//filesystem::FileSystem fs;

	for (auto& path : std::filesystem::directory_iterator(input_path))
	{
		if (std::filesystem::is_directory(path))
		{
			std::cout << path << std::endl;

			std::vector<std::string> allCSVs;

			for (auto algorithmsFolders : std::filesystem::directory_iterator((path / "1")))
			{
				if (std::filesystem::is_directory(algorithmsFolders))
				{
					if (algorithmsFolders.path().string().find("EnsembleList_DistanceScheme") != std::string::npos
						|| algorithmsFolders.path().string().find("EnsembleList_DistanceScheme") != std::string::npos)
					{
						allCSVs = getAllCsvs(algorithmsFolders / "regions");
						break;
					}
				}
			}

			auto trainCertain = filterOut(allCSVs, "CertainTrain");
			auto uncertainTrain = filterOut(allCSVs, "uncertainTrain");

			auto certainVAL = filterOut(allCSVs, "CertainVAL");
			auto uncertainVAL = filterOut(allCSVs, "uncertainVAL");

			auto certainTEST = filterOut(allCSVs, "CertainTEST");
			auto uncertainTEST = filterOut(allCSVs, "uncertainTEST");

			std::sort(trainCertain.begin(), trainCertain.end(), compareNat);
			std::sort(uncertainTrain.begin(), uncertainTrain.end(), compareNat);
			std::sort(certainVAL.begin(), certainVAL.end(), compareNat);
			std::sort(uncertainVAL.begin(), uncertainVAL.end(), compareNat);
			std::sort(certainTEST.begin(), certainTEST.end(), compareNat);
			std::sort(uncertainTEST.begin(), uncertainTEST.end(), compareNat);

			std::vector<std::filesystem::path> algorithms;

			for (auto folder :  std::filesystem::directory_iterator(path))
			{


				std::string algorithmPath((path / folder).string());

				std::shared_ptr<phd::svm::ISvm> svm;

				for (auto& path : std::filesystem::directory_iterator(algorithmPath))
				{
					if (path.is_directory())
					{
						auto algorithmName = platform::stringUtils::splitString(path.path().filename().string(), "__")[0];
						auto svmPath = getAllSvms(path);

						if (algorithmName.find("EnsembleList_With_Alma") != std::string::npos || algorithmName.find("EnsembleList_With_Alma") != std::string::
							npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], true);
							//svm = std::make_shared<phd::svm::libSvmImplementation>(svmPath[0]);
						}
						else if (algorithmName.find("EnsembleList_DistanceScheme") != std::string::npos || algorithmName.find("EnsembleList_DistanceScheme") !=
							std::string::npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], false);
						}
						else if (algorithmName.find("EnsembleList_ALMA_no_inheritance") != std::string::npos || algorithmName.find(
							"EnsembleList_ALMA_no_inheritance") != std::string::npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], true);
						}
						else
						{
							svm = std::make_shared<phd::svm::libSvmImplementation>(svmPath[0]);
						}

						std::ofstream output(algorithmPath + "\\" + algorithmName + "_results.txt");

						for (auto i = 0u; i < trainCertain.size(); ++i)
						{
							auto normalize = false;
							auto certainLoader = genetic::LocalFileDatasetLoader(trainCertain[i], certainVAL[i], certainTEST[i], normalize);
							auto uncertainLoader = genetic::LocalFileDatasetLoader(uncertainTrain[i], uncertainVAL[i], uncertainTEST[i], normalize);

							svmComponents::SvmAccuracyMetric acc;
							svmComponents::BaseSvmChromosome individual;
							individual.updateClassifier(svm);
							auto resultTest = acc.calculateMetric(individual, certainLoader.getTestSet(), true);

							//svmComponents::SvmVisualization svmVis;
							//auto result = svmVis.createDetailedVisualization(*svm, 500, 500, certainLoader.getTraningSet(), certainLoader.getValidationSet(), certainLoader.getTestSet());
							//std::filesystem::path m_pngNameSource;
							//genetic::SvmWokrflowConfiguration config_copy5{ "", "", "", R"(D:\ENSEMBLE_CLASSIFICATION_412_LASTNODE_FULL_SVM_2D_REGIONS\\)", "debug_" + std::to_string(i), "" };
							//setVisualizationFilenameAndFormat(svmComponents::imageFormat::png, m_pngNameSource, config_copy5);
							//strategies::FileSinkStrategy m_savePngElement;
							//m_savePngElement.launch(result, m_pngNameSource);

							output << resultTest.m_confusionMatrix.value() << "\t";

							//std::cout << "ID: " << i << "  " << resultTest.m_confusionMatrix << "\n";


							//TODO better handle empty sets in dataset loader
							if (!uncertainLoader.getTestSet().empty())
							{
								auto resultuncertainTest = acc.calculateMetric(individual, uncertainLoader.getTestSet(), true);
								output << resultuncertainTest.m_confusionMatrix.value() << "\n";
							}
							else
							{
								output << svmComponents::ConfusionMatrix(0, 0, 0, 0) << "\n";
							}
						}
					}
				}
			}
		}
	}
}


inline std::vector<std::string> listDir(std::string& path)
{
	std::vector<std::string> result;
	//filesystem::FileSystem fs;
	for (auto& path :  std::filesystem::directory_iterator(path))
	{
		if (std::filesystem::is_directory(path))
		{
			result.emplace_back(path.path().string());
		}
	}
	return result;
}

inline void handleUncertain(std::ofstream& output, const dataset::Dataset<std::vector<float>, float>& data, svmComponents::BaseSvmChromosome individual)
{
	svmComponents::SvmAccuracyMetric acc;

	if (!data.empty())
	{
		auto resultuncertainTest = acc.calculateMetric(individual, data, true, true);
		output << resultuncertainTest.m_confusionMatrix.value() << "\t";

		std::cout << "Uncertain:" << resultuncertainTest.m_confusionMatrix.value() << "\n";
	}
	else
	{
		output << svmComponents::ConfusionMatrix(0, 0, 0, 0) << "\t";
	}
}


inline void handleCertain(std::ofstream& output, const dataset::Dataset<std::vector<float>, float>& data, svmComponents::BaseSvmChromosome individual)
{
	svmComponents::SvmAccuracyMetric acc;

	if (!data.empty())
	{
		auto resultuncertainTest = acc.calculateMetric(individual, data, true, true);
		output << resultuncertainTest.m_confusionMatrix.value() << "\t";

		std::cout << "Certain:" << resultuncertainTest.m_confusionMatrix.value() << "\n";
	}
	else
	{
		output << svmComponents::ConfusionMatrix(0, 0, 0, 0) << "\t";
	}
}


class RegionsScores
{
public:
	RegionsScores(int argc, char* argv[])
	{
		m_config = testApp::parseCommandLineArguments(argc, argv);
	}

	std::shared_ptr<phd::svm::ISvm> loadSvm(const std::string& svmPath, const std::string& algorithmName)
	{
		std::shared_ptr<phd::svm::ISvm> svm;

		if (algorithmName.find("EnsembleList_With_Alma") != std::string::npos || algorithmName.find("EnsembleList_With_Alma") != std::string::
			npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
		}
		else if (algorithmName.find("EnsembleList_DistanceScheme") != std::string::npos || algorithmName.find("EnsembleList_DistanceScheme") !=
			std::string::npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, false);
		}
		else if (algorithmName.find("EnsembleList_ALMA_no_inheritance") != std::string::npos || algorithmName.find(
			"EnsembleList_ALMA_no_inheritance") != std::string::npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
		}
		else if (algorithmName.find("Baseline_EnsembleList_RBF") != std::string::npos || algorithmName.find(
			"Baseline_EnsembleList_RBF") != std::string::npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, false);
		}
		else
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
		//	svm = std::make_shared<phd::svm::libSvmImplementation>(svmPath);
		}

		return svm;
	}

	void createOutput(std::shared_ptr<phd::svm::ISvm> svm,
	                  std::ofstream& output,
	                  double avg_sv,
	                  genetic::LocalFileDatasetLoader& certainLoader,
	                  genetic::LocalFileDatasetLoader& uncertainLoader,
	                  svmComponents::BaseSvmChromosome& individual)
	{
		handleCertain(output, certainLoader.getTestSet(), individual);
		handleUncertain(output, uncertainLoader.getTestSet(), individual);
		handleCertain(output, certainLoader.getValidationSet(), individual);
		handleUncertain(output, uncertainLoader.getValidationSet(), individual);
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

	double getSvNumber(std::shared_ptr<phd::svm::ISvm> svm)
	{
		double avg_sv = 0.0;
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
		return avg_sv;
	}

	void getAllRegions(std::vector<std::string> allCSVs, std::vector<std::string>& trainCertain, std::vector<std::string>& uncertainTrain, std::vector<std::string>& certainVAL, std::vector<std::string>& uncertainVAL, std::vector<std::string>& certainTEST, std::vector<std::string>& uncertainTEST)
	{
		trainCertain = filterOut(allCSVs, "CertainTrain");
		uncertainTrain = filterOut(allCSVs, "uncertainTrain");
		certainVAL = filterOut(allCSVs, "CertainVAL");
		uncertainVAL = filterOut(allCSVs, "uncertainVAL");
		certainTEST = filterOut(allCSVs, "CertainTEST");
		uncertainTEST = filterOut(allCSVs, "uncertainTEST");

		std::sort(trainCertain.begin(), trainCertain.end(), compareNat);
		std::sort(uncertainTrain.begin(), uncertainTrain.end(), compareNat);
		std::sort(certainVAL.begin(), certainVAL.end(), compareNat);
		std::sort(uncertainVAL.begin(), uncertainVAL.end(), compareNat);
		std::sort(certainTEST.begin(), certainTEST.end(), compareNat);
		std::sort(uncertainTEST.begin(), uncertainTEST.end(), compareNat);
	}

	void resulstForSingleRun(std::filesystem::path fold, std::filesystem::path algorithm, std::string algorithmPath, std::vector<std::string> allCSVs)
	{
		allCSVs = getAllCsvs(algorithmPath / std::filesystem::path("regions"));

		std::vector<std::string> trainCertain;
		std::vector<std::string> uncertainTrain;
		std::vector<std::string> certainVAL;
		std::vector<std::string> uncertainVAL;
		std::vector<std::string> certainTEST;
		std::vector<std::string> uncertainTEST;
		getAllRegions(allCSVs, trainCertain, uncertainTrain, certainVAL, uncertainVAL, certainTEST, uncertainTEST);

		auto algorithmName = platform::stringUtils::splitString(algorithm.stem().string(), "__")[0];
		auto svmPath = getAllSvms(algorithmPath);

		std::cout << algorithmPath << "\n";
		std::cout << algorithmName << "\n";

		std::shared_ptr<phd::svm::ISvm> svm = loadSvm(svmPath[0].string(), algorithmName);

		auto avg_sv = getSvNumber(svm);

		for (auto i = trainCertain.size() - 1; i < trainCertain.size(); ++i)
		{
			auto normalize = false;
			auto resample = false;
			auto certainLoader = genetic::LocalFileDatasetLoader(trainCertain[i], certainVAL[i], certainTEST[i], normalize, resample);
			auto uncertainLoader = genetic::LocalFileDatasetLoader(uncertainTrain[i], uncertainVAL[i], uncertainTEST[i], normalize,
			                                                       resample);

			svmComponents::BaseSvmChromosome individual;
			individual.updateClassifier(svm);

			std::ofstream output(fold.string() + "\\" + algorithmName + "_LastNode.txt");
 			createOutput(svm, output, avg_sv, certainLoader, uncertainLoader, individual);
		}
	}

	void runAlgorithm()
	{
		//auto input_path = R"(D:\ENSEMBLE_631_2D_old_ET_V_ClassBias)";
		auto input_path = m_config.outputFolder;
		//filesystem::FileSystem fs;

		for (auto& dataset :  std::filesystem::directory_iterator(input_path))
		{
			if (std::filesystem::is_directory(dataset))
			{
				for (auto fold :  std::filesystem::directory_iterator(dataset))
				{
					if (!std::filesystem::is_directory(fold))
						continue;
					 
					//folders with algorithms in here
					for (auto algorithm :  std::filesystem::directory_iterator(fold))
					{
						std::string algorithmPath((dataset / algorithm).string());
						std::vector<std::string> allCSVs;
						if (std::filesystem::is_directory(algorithmPath))
						{
							resulstForSingleRun(fold, algorithm, algorithmPath, allCSVs);
						}
					}
				}
			}
		}
	}
	
private:
	testApp::configTestApp m_config;
};


inline void lastRegionsScoreExperiment(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);

	//auto input_path = config.outputFolder;
	//auto input_path = R"(D:\ENSEMBLE_CLASSIFICATION_413_LASTNODE_FULL_SVM_NON2D_REGIONS)";
	auto input_path = R"(D:\GECCO_Results\2D_results\2D_ENSEMBLE_810_GECCO_final_2D2)";


	for (auto& path :  std::filesystem::directory_iterator(input_path))
	{
		if (std::filesystem::is_directory(path))
		{
			std::cout << path << std::endl;
			std::vector<std::string> allCSVs;

			for (auto folder :  std::filesystem::directory_iterator(path))
			{
				if (!std::filesystem::is_directory(folder))
					continue;
				//auto folds = listDir(path.string());

				for (auto folder_deeper :  std::filesystem::directory_iterator(folder))
				{

					std::string regionsPath((path / folder_deeper).string());

					if (std::filesystem::is_directory(regionsPath))
					{
						if (regionsPath.find("EnsembleList_With_Alma__") != std::string::npos)
						//if (regionsPath.find("EnsembleList_With_Alma_NF_Baseline_V") != std::string::npos)
						{
							allCSVs = getAllCsvs(regionsPath / std::filesystem::path("regions"));
							break;
						}
					}
				}

				auto trainCertain = filterOut(allCSVs, "CertainTrain");
				auto uncertainTrain = filterOut(allCSVs, "uncertainTrain");

				auto certainVAL = filterOut(allCSVs, "CertainVAL");
				auto uncertainVAL = filterOut(allCSVs, "uncertainVAL");

				auto certainTEST = filterOut(allCSVs, "CertainTEST");
				auto uncertainTEST = filterOut(allCSVs, "uncertainTEST");

				std::sort(trainCertain.begin(), trainCertain.end(), compareNat);
				std::sort(uncertainTrain.begin(), uncertainTrain.end(), compareNat);
				std::sort(certainVAL.begin(), certainVAL.end(), compareNat);
				std::sort(uncertainVAL.begin(), uncertainVAL.end(), compareNat);
				std::sort(certainTEST.begin(), certainTEST.end(), compareNat);
				std::sort(uncertainTEST.begin(), uncertainTEST.end(), compareNat);


				std::string algorithmPath((path / folder).string());
				std::shared_ptr<phd::svm::ISvm> svm;


				for (auto& path : std::filesystem::directory_iterator(algorithmPath))
				{
					if (path.is_directory())
					{
						auto algorithmName = platform::stringUtils::splitString(path.path().filename().string(), "__")[0];
						auto svmPath = getAllSvms(path);

						if(svmPath.empty())
						{
							continue;
						}

						std::cout << path.path() << "\n";
						std::cout << algorithmName << "\n";


						if (algorithmName.find("EnsembleList_With_Alma") != std::string::npos || algorithmName.find("EnsembleList_With_Alma") != std::string::
							npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], true);
						}
						else if (algorithmName.find("EnsembleList_DistanceScheme") != std::string::npos || algorithmName.find("EnsembleList_DistanceScheme") !=
							std::string::npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], false);
						}
						else if (algorithmName.find("EnsembleList_ALMA_no_inheritance") != std::string::npos || algorithmName.find(
							"EnsembleList_ALMA_no_inheritance") != std::string::npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], true);
						}
						else if (algorithmName.find("Baseline_EnsembleList_RBF") != std::string::npos || algorithmName.find(
							"Baseline_EnsembleList_RBF") != std::string::npos)
						{
							svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[0], false);
						}
						else
						{
							svm = std::make_shared<phd::svm::libSvmImplementation>(svmPath[0]);
						}

						std::ofstream output(algorithmPath + "\\" + algorithmName + "_LastNode.txt");

						std::vector<float> test_percent;

						for (auto i = 0u; i < trainCertain.size(); ++i)
						{
							/*auto normalize = false;
							auto certainLoader = genetic::LocalFileDatasetLoader(trainCertain[i], certainVAL[i], certainTEST[i], normalize);
							auto uncertainLoader = genetic::LocalFileDatasetLoader(uncertainTrain[i], uncertainVAL[i], uncertainTEST[i], normalize);

							test_percent.emplace_back(static_cast<float>(certainLoader.getTestSet().size()) / static_cast<float>(certainLoader.getTestSet().size() + uncertainLoader.getTestSet().size()));

							if(auto last_number = trainCertain.size() - 1; i == last_number )
							{
								test_percent.emplace_back(static_cast<float>(uncertainLoader.getTestSet().size()) / static_cast<float>(certainLoader.getTestSet().size() + uncertainLoader.getTestSet().size()));
							}*/

						}

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

						for (auto i = trainCertain.size() - 1; i < trainCertain.size(); ++i)
						{
							auto normalize = false;
							auto resample = false;

							auto certainLoader = genetic::LocalFileDatasetLoader(trainCertain[i], certainVAL[i], certainTEST[i], normalize, resample);
							auto uncertainLoader = genetic::LocalFileDatasetLoader(uncertainTrain[i], uncertainVAL[i], uncertainTEST[i], normalize, resample);

							svmComponents::SvmAccuracyMetric acc;
							svmComponents::BaseSvmChromosome individual;
							individual.updateClassifier(svm);
							//auto resultTest = acc.calculateMetric(individual, certainLoader.getTestSet(), true, true);
							//auto resultVal = acc.calculateMetric(individual, certainLoader.getValidationSet(), true, true);
							//auto resultTr = acc.calculateMetric(individual, certainLoader.getTraningSet(), true, true);

						/*	auto train = R"(D:\journal_datasets\australian\3\train.csv)";
							auto val = R"(D:\journal_datasets\australian\3\validation.csv)";
							auto test = R"(D:\journal_datasets\australian\3\test.csv)";
							auto full = genetic::LocalFileDatasetLoader(train, val, test);*/


							//auto resultTestFull = acc.calculateMetric(individual, full.getTestSet(), true, true);

							//std::cout << "Full:" << resultTestFull.m_confusionMatrix << "\n";
							//std::cout << "Certain:" << resultTest.m_confusionMatrix << "\n";

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
					}
				}
			}
		}
	}
}
