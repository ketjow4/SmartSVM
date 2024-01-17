#include "libPlatform/loguru.hpp"
#include "libPlatform/StringUtils.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "SvmEnsembleHelper.h"
#include "EnsembleTreeWorkflow.h"


#include "DefaultWorkflowConfigs.h"
#include "GridSearchWorkflow.h"
#include "libRandom/MersenneTwister64Rng.h"
#include "SvmLib/libSvmImplementation.h"
#include "SvmLib/EnsembleListSvm.h"
#include "libSvmComponents/LinearKernel.h"
#include "libSvmComponents/RbfKernel.h"
#include "libGeneticSvm/DatasetLoaderHelper.h"
#include "libSvmComponents/SvmAccuracyMetric.h"
#include "SvmAlgorithmFactory.h"
#include "EnsembleUtils.h"

namespace genetic
{

	dataset::Dataset<std::vector<float>, float> processUncertain(const dataset::Dataset<std::vector<float>, float>& dataset, phd::svm::EnsembleListSvm trained_one)
	{
		std::vector<uint64_t> ids(dataset.size());
		std::iota(ids.begin(), ids.end(), 0);
		auto [uncertainTestSet, uncertainIdsTest] = getUncertainDataset(dataset, ids, trained_one.root->m_svm);
		svmComponents::SvmTrainingSetChromosome uncertainTest{ std::move(uncertainTestSet) };
		auto filtered_test_ = uncertainTest.convertChromosome(dataset);
		return filtered_test_;
	}


	std::tuple<dataset::Dataset<std::vector<float>, float>, std::vector<unsigned long long>,
			   dataset::Dataset<std::vector<float>, float>, std::vector<unsigned long long>>
	EnsembleTreeWorkflow::resample(const dataset::Dataset<std::vector<float>, float>& dataset,
	                               const std::vector<unsigned long long>& ids,
	                               const dataset::Dataset<std::vector<float>, float>& validationSet,
	                               const std::vector<unsigned long long>& valIds)
	{

		
		

		
		auto joined_tr_val = joinSets(dataset, validationSet);
		std::vector<unsigned long long>  joined_ids;
		joined_ids.insert(joined_ids.end(), ids.begin(), ids.end());
		joined_ids.insert(joined_ids.end(), valIds.begin(), valIds.end());
		
		std::unordered_set<uint64_t> indexsSet;
		dataset::Dataset<std::vector<float>, float> newValidation;
		std::vector<unsigned long long>  newValidationIds;

		auto rngEngine = std::make_unique<random::MersenneTwister64Rng>(0);
		auto randomID = std::uniform_int_distribution<int>(0, static_cast<int>(joined_tr_val.size() - 1));


		auto classCount = svmUtils::countLabels(2, joined_tr_val);

		int minTrainingSizeForClass = m_algorithmConfig.m_trainingSetOptimization->getInitialTrainingSetSize();
		int increasedTrainingSize = 8;

		
		if (!m_full_config.getValue<bool>("Svm.EnsembleTree.ResamplingWithNoAddition"))
		{
			long class0 = classCount[0];
			long class1 = classCount[1];
			
			if (m_full_config.getValue<bool>("Svm.EnsembleTree.NewSamplesForTraining"))
			{
				//make sure that there will be enough vectors to run genetic algorithm
				class0 -= increasedTrainingSize * minTrainingSizeForClass;
				class1 -= increasedTrainingSize * minTrainingSizeForClass;
			}

			auto negativeNumber = class0;
			auto data = m_joined_T_V;
			auto randomVector = std::uniform_int_distribution<int>(0, static_cast<int>(data.size() - 1));

			while (negativeNumber < static_cast<long>(m_validationNegative))
			{
				
				auto index = rngEngine->getRandom(randomVector);

				if (std::find(joined_ids.begin(), joined_ids.end(), index) == joined_ids.end() && data.getLabel(index) == 0)
				{
					joined_ids.emplace_back(index);
					joined_tr_val.addSample(data.getSample(index), data.getLabel(index));
					negativeNumber++;
				}
			}

			auto positiveNumber = class1;
			while (positiveNumber < static_cast<long>(m_validationPositive))
			{
				

				auto index = rngEngine->getRandom(randomVector);

				if (std::find(joined_ids.begin(), joined_ids.end(), index) == joined_ids.end() && data.getLabel(index) == 1)
				{
					joined_ids.emplace_back(index);
					joined_tr_val.addSample(data.getSample(index), data.getLabel(index));
					positiveNumber++;
				}
			}
		}
		else
		{
			//make sure that there will be enough vectors to run genetic algorithm
			if(static_cast<int>(classCount[0] - m_validationNegative) < minTrainingSizeForClass)
			{
				m_validationNegative -= increasedTrainingSize * minTrainingSizeForClass - (classCount[0] - m_validationNegative);
			}
			if (static_cast<int>(classCount[1] - m_validationPositive) < minTrainingSizeForClass)
			{
				m_validationPositive -= increasedTrainingSize * minTrainingSizeForClass - (classCount[1] - m_validationPositive);
			}
		}

		auto maxNumberOfTries = 100000;
		auto tries = 0;
		unsigned int numberOfExamples = 0;
		while (numberOfExamples < m_validationNegative && tries < maxNumberOfTries)
		{
			auto index = rngEngine->getRandom(randomID);
			if (joined_tr_val.getLabel(index) == 0
				&& indexsSet.emplace(static_cast<int>(index)).second)
			{
				newValidation.addSample(joined_tr_val.getSample(index), joined_tr_val.getLabel(index));
				newValidationIds.emplace_back(joined_ids[index]);
				numberOfExamples++;
			}
			else
			{
				tries++;
			}
		}

		numberOfExamples = 0;
		tries = 0;
		while (numberOfExamples < m_validationPositive && tries < maxNumberOfTries)
		{
			auto index = rngEngine->getRandom(randomID);
			if (joined_tr_val.getLabel(index) == 1
				&& indexsSet.emplace(static_cast<int>(index)).second)
			{
				newValidation.addSample(joined_tr_val.getSample(index), joined_tr_val.getLabel(index));
				newValidationIds.emplace_back(joined_ids[index]);
				numberOfExamples++;
			}
			else
			{
				tries++;
			}
		}


		dataset::Dataset<std::vector<float>, float> newTraining;
		std::vector<unsigned long long>  newTrainingIds;
		for (auto i = 0u; i < joined_tr_val.size(); ++i)
		{
			if (indexsSet.emplace(static_cast<int>(i)).second)
			{
				newTraining.addSample(joined_tr_val.getSample(i), joined_tr_val.getLabel(i));
				newTrainingIds.emplace_back(joined_ids[i]);
			}
		}

		LOG_F(INFO, "Resampling size of training %d, size of validation %d", newTraining.size(), newValidation.size());

		return std::make_tuple(newTraining, newTrainingIds, newValidation, newValidationIds);
	}


EnsembleTreeWorkflow::EnsembleTreeWorkflow(const SvmWokrflowConfiguration& config,
                                           EnsembleTreeWorkflowConfig algorithmConfig,
                                           IDatasetLoader& workflow,
                                           platform::Subtree full_config)
	: m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
	, m_algorithmConfig(algorithmConfig)
	, m_config(config)
	, m_loadingWorkflow(&workflow)
	, m_full_config(full_config)
	, m_metric(svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc))
	, m_validation(*m_metric, false)
	, m_validationTest(*m_metric, true)
	, m_listLength(0)
	, m_id(0)
	, m_newClassificationScheme(true) //TODO extract to config
	, m_useDasvmKernel(algorithmConfig.m_useDasvmKernel)
	, m_debugLog(full_config.getValue<bool>("Svm.EnsembleTree.DebugLog"))
	, m_useFeatureSelection(full_config.getValue<bool>("Svm.EnsembleTree.UseFeatureSelction"))
{

	m_joined_T_V = joinSets(m_loadingWorkflow->getTraningSet(), m_loadingWorkflow->getValidationSet());
}

void EnsembleTreeWorkflow::createVisualization(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                               const dataset::Dataset<std::vector<float>, float>& validationSet,
                                               int& id_,
                                               std::shared_ptr<phd::svm::ISvm> svm,
                                               const std::vector<uint64_t>& trIds,
                                               const std::vector<uint64_t>& valIds)
{
	std::filesystem::path m_pngNameSource;
	svmComponents::SvmVisualization visualization3;
	auto image3 = visualization3.createEnsembleVisualization(*svm, 500, 500, trainingSet, validationSet, m_loadingWorkflow->getTestSet());;
	SvmWokrflowConfiguration config_copy3{"", "", "", m_config.outputFolderPath, "m_listNode", ""};
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, id_);
	m_savePngElement.launch(image3, m_pngNameSource);

	auto [uncertainValidationSet, uncertainIdsValidation] = getUncertainDataset(validationSet, valIds, svm);
	svmComponents::SvmTrainingSetChromosome uncertainValidation{std::move(uncertainValidationSet)};
	auto finalResult = uncertainValidation.convertChromosome(m_loadingWorkflow->getValidationSet());

	auto new_set = visualization3.createVisualizationNewValidationSet(500, 500, finalResult);
	SvmWokrflowConfiguration config_copy4{"", "", "", m_config.outputFolderPath, "shrinkedValidation", ""};
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy4, id_);
	m_savePngElement.launch(new_set, m_pngNameSource);

	auto [uncertainTrainingSet, uncertainIds] = getUncertainDataset(trainingSet, trIds, svm);

	auto [svDatasetVectors, svIds] = getSVsIds(trainingSet, trIds, svm);
	uncertainTrainingSet.insert(uncertainTrainingSet.end(), svDatasetVectors.begin(), svDatasetVectors.end());
	uncertainIds.insert(uncertainIds.end(), svIds.begin(), svIds.end());

	svmComponents::SvmTrainingSetChromosome uncertainTraining{std::move(uncertainTrainingSet)};
	auto finalResultTR = uncertainTraining.convertChromosome(m_loadingWorkflow->getTraningSet());

	auto new_setTR = visualization3.createVisualizationNewValidationSet(500, 500, finalResultTR);
	SvmWokrflowConfiguration config_copy5{"", "", "", m_config.outputFolderPath, "shrinkedTrainingSet", ""};
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy5, id_);
	m_savePngElement.launch(new_setTR, m_pngNameSource);
}

	void EnsembleTreeWorkflow::createVisualizationNewFlow(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                      const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                                      int& id_,
	                                                      std::shared_ptr<phd::svm::ISvm> svm,
	                                                      const std::vector<uint64_t>& /*trIds*/,
	                                                      const std::vector<uint64_t>& /*valIds*/)
{
	std::filesystem::path m_pngNameSource;
	svmComponents::SvmVisualization visualization3;
	auto image3 = visualization3.createEnsembleVisualization(*svm, 500, 500, trainingSet, validationSet, m_loadingWorkflow->getTestSet());;
	SvmWokrflowConfiguration config_copy3{ "", "", "", m_config.outputFolderPath, "m_listNode", "" };
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3, id_);
	m_savePngElement.launch(image3, m_pngNameSource);
}



std::vector<std::filesystem::path> getSvms(std::filesystem::path folderPath)
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

void EnsembleTreeWorkflow::addLastNodeWithFullSvm(std::shared_ptr<phd::svm::ListNodeSvm>& temp)
{

	if(m_full_config.getValue<bool>("Svm.EnsembleTree.AddAlmaNode"))
	{
		//Load pre-trained ALMA classifier
		//std::shared_ptr<phd::svm::ISvm> svmLast;
		//auto outputfolderName = m_full_config.getValue<std::string>("Svm.OutputFolderPath");
		//filesystem::FileSystem fs;

		//for(auto path : fs.directoryIterator(outputfolderName + "\\.."))
		//{
		//	std::string name("ALMA");
		//	//std::string name("SE-SVM");
		//	if(path.string().find(name) != std::string::npos && fs.isDirectory(path))
		//	{
		//		auto svmList = getSvms(path);
		//		assert(svmList.size() == 1 && "didn't find proper svm");

		//		svmLast = std::make_shared<phd::svm::libSvmImplementation>(svmList[0]);

		//		break;
		//	}
		//}

		
		////here ALMA run from scratch  algorithm
		//auto almaConfig = genetic::DefaultAlgaConfig::getALMA();

		//almaConfig.putValue<std::string>("Svm.Metric", "AUC");
		////setupStopCondition(almaConfig);
		//auto outputfolderName = m_full_config.getValue<std::string>("Svm.OutputFolderPath");
		//std::cout << "output folder " << outputfolderName << "\n";
		//almaConfig.putValue<std::string>("Svm.OutputFolderPath", outputfolderName);
		//almaConfig.putValue<std::string>("Svm.TxtLogFilename", "alma_log.txt");
		//genetic::SvmAlgorithmFactory fac;

		//auto al = fac.createAlgorightm(almaConfig, *m_loadingWorkflow);
		//auto svmLast = al->run();
		//
		
		//SE-SVM as last node
		auto sesvmConfig = genetic::DefaultSSVMConfig::getDefault();

		sesvmConfig.putValue<std::string>("Svm.Metric", "AUC");
		//setupStopCondition(almaConfig);
		auto outputfolderName = m_full_config.getValue<std::string>("Svm.OutputFolderPath");
		std::cout << "output folder " << outputfolderName << "\n";
		sesvmConfig.putValue<std::string>("Svm.OutputFolderPath", outputfolderName);
		sesvmConfig.putValue<std::string>("Svm.TxtLogFilename", "sesvm_log.txt");
		sesvmConfig.putValue<std::string>("Svm.ValidationData", m_full_config.getValue<std::string>("Svm.ValidationData"));
		sesvmConfig.putValue<std::string>("Svm.TestData", m_full_config.getValue<std::string>("Svm.TestData"));
		sesvmConfig.putValue<std::string>("Svm.TrainingData", m_full_config.getValue<std::string>("Svm.TrainingData"));


		auto numberOfFeatures = m_loadingWorkflow->getTraningSet().getSample(0).size();
		if(numberOfFeatures > 4)
		{
			sesvmConfig.putValue("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", 4);
		}
		else
		{
			sesvmConfig.putValue("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", numberOfFeatures);
		}
		
		genetic::SvmAlgorithmFactory fac;

		auto al = fac.createAlgorightm(sesvmConfig, *m_loadingWorkflow);
		auto svmLast = al->run();

		
		temp->m_next = std::make_shared<phd::svm::ListNodeSvm>();
		temp = temp->m_next;
		temp->m_svm = svmLast;

		if(m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			std::filesystem::path m_pngNameSource;
			svmComponents::SvmVisualization visualization3;
			auto image3 = visualization3.createDetailedVisualization(*svmLast, 500, 500, m_loadingWorkflow->getTraningSet(), m_loadingWorkflow->getValidationSet(), m_loadingWorkflow->getTestSet());;
			SvmWokrflowConfiguration config_copy3{ "", "", "", m_config.outputFolderPath, "ALMA_LAST_NODE", "" };
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
			m_savePngElement.launch(image3, m_pngNameSource);
		}
		m_newClassificationScheme = true;
	}
	else
	{
		//TODO fix after inital experiment with regions
		m_newClassificationScheme = false;
	}
}






std::shared_ptr<phd::svm::ListNodeSvm> EnsembleTreeWorkflow::trainHelperNewDatasetFlow(std::shared_ptr<phd::svm::ListNodeSvm>& root_,
	                                                                                       const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                                                       const std::vector<uint64_t>& ids,
	                                                                                       const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                                                                       const std::vector<uint64_t>& valIds)
{
	//!!!!!!!!!!!!!! RESAMPLE training set --> train, validation
	auto [tr_l, trIds, val_l, validationIDS] = resample(trainingSet, ids, validationSet, valIds);

	
	try
	{
		const int max_length = 250;
		//const int max_length = 3;
		const int numberOfClasses = 2;
		const int min_k = m_algorithmConfig.m_trainingSetOptimization->getInitialTrainingSetSize();

		std::shared_ptr<phd::svm::ISvm> svm;
		std::shared_ptr<phd::svm::ISvm> last_svm;
		bool datasetTooSmall = false;
		bool nothingImprovedInValidation = false;
		bool nothingImprovedInTraining = false;
		bool nothingImprovedInFullSet = false;
		bool done = false;
		bool doneTR = false;
	/*	dataset::Dataset<std::vector<float>, float> tr_l = trainingSet;
		dataset::Dataset<std::vector<float>, float> val_l = validationSet;*/
		dataset::Dataset<std::vector<float>, float> test_l = m_loadingWorkflow->getTestSet();
		//std::vector<uint64_t> trIds = ids;
		//std::vector<uint64_t> validationIDS = valIds;

		dataset::Dataset<std::vector<float>, float> fullSet = trainingSet;
		std::vector<uint64_t> fullSetIds = ids;
		
		std::vector<uint64_t> svIdsGlobal;
		std::vector<DatasetVector> svVectorsGlobal;
		std::vector<DatasetVector> svVectorsToPassDown;

		std::vector<DatasetVector> previousUncertain;

		std::vector<Gene> svVectorsWithGammas;
		std::vector<Gene> svVectorsWithGammasGlobalPool;
		std::unordered_set<uint64_t> svVectorsWithGammaIdSet;

		std::vector<uint64_t> testSetIds;
		for (auto i = 0u; i < m_loadingWorkflow->getTestSet().size(); ++i)
		{
			testSetIds.emplace_back(i);
		}

		std::vector<uint64_t> countTR = countClasses(trainingSet);

		auto temp = root_;
		auto stopCondition = [&]()
		{
			return (nothingImprovedInFullSet || m_id >= max_length || datasetTooSmall);
		};

		while (!stopCondition()) //limit size of ensemble for visualization speed problems
		{
			bool linearIsBetter = false;
			DatasetLoaderHelper datasets(tr_l, val_l, m_loadingWorkflow->getTestSet());
			{
				try
				{
					
				
				std::cout << "ID:" << m_listLength << "\n";
				auto configurationForNode = EnsembleTreeWorkflowConfig(m_full_config, datasets);

				bool newDatasetFlow = true;
				if (m_full_config.getValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet"))
				{
					newDatasetFlow = false; //see SvmHelper Evaluate function to see wh it is false
				}
				
					
				SvmHelper helper(m_config, configurationForNode, datasets, m_algorithmConfig.m_addSvToTraining,
					svVectorsToPassDown, *m_loadingWorkflow, svVectorsWithGammas,
					m_useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config, newDatasetFlow);
				svm = helper.run();


				SvmHelper helper4(m_config, configurationForNode, datasets,false,
					svVectorsToPassDown, *m_loadingWorkflow, svVectorsWithGammas,
					false, m_debugLog, m_useFeatureSelection, m_full_config, newDatasetFlow);
				auto svm_no_inheritance = helper4.run();


					

				auto config_copy = m_full_config;
				config_copy.putValue("Svm.KernelType", "LINEAR");
				config_copy.putValue("Svm.EnsembleTree.DasvmKernel", false);
				auto useDasvmKernel = false;
				auto configurationForNodeLinear = EnsembleTreeWorkflowConfig(config_copy, datasets);
				SvmHelper helper2(m_config, configurationForNodeLinear, datasets, false,
					svVectorsToPassDown, *m_loadingWorkflow, svVectorsWithGammas,
					useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config, newDatasetFlow);
				auto linearsvm = helper2.run();



				/*config_copy.putValue("Svm.KernelType", "POLY");
				config_copy.putValue("Svm.EnsembleTree.DasvmKernel", false);
				auto configurationForNodePoly = EnsembleTreeWorkflowConfig(config_copy, datasets);
				SvmHelper helper3(m_config, configurationForNodePoly, datasets, false,
					svVectorsToPassDown, m_loadingWorkflow, svVectorsWithGammas,
					useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config, newDatasetFlow);
				auto polysvm = helper3.run();*/

				if (helper4.getBestOne().getFitness() > helper.getBestOne().getFitness())
				{
					LOG_F(INFO, "No inheritance is better %.5f  RBF: %.5f", helper4.getBestOne().getFitness(), helper.getBestOne().getFitness());
					svm = svm_no_inheritance;
					linearIsBetter = true;

					if (helper2.getBestOne().getFitness() > helper4.getBestOne().getFitness())
					{
						LOG_F(INFO, "Linear is better %.5f  RBF: %.5f", helper2.getBestOne().getFitness(), helper.getBestOne().getFitness());
						svm = linearsvm;
						linearIsBetter = true;
					}
					
				}
				else if(helper2.getBestOne().getFitness() > helper.getBestOne().getFitness())
				{
					LOG_F(INFO, "Linear is better %.5f  RBF: %.5f", helper2.getBestOne().getFitness(), helper.getBestOne().getFitness());
					svm = linearsvm;
					linearIsBetter = true;
				}

				/*if (helper3.getBestOne().getFitness() > helper.getBestOne().getFitness() && helper3.getBestOne().getFitness() > helper2.getBestOne().getFitness())
				{
					LOG_F(INFO, "Poly is better %.5f  RBF: %.5f", helper3.getBestOne().getFitness(), helper.getBestOne().getFitness());
					svm = polysvm;
					linearIsBetter = true;
				}*/
					
				//auto tempNone = std::shared_ptr<phd::svm::ISvm>();
				//svm = runGridSearch(datasets, tempNone);
				//
				temp->m_svm = svm;
				}
				catch (const std::exception& e)
				{
					LOG_F(ERROR, "Error: %s", e.what());
					std::cout << e.what();
					throw;
				}

			}

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				if (m_useDasvmKernel && !previousUncertain.empty())
				{
					std::vector<DatasetVector> svWithOtherIds;

					int idCounter = 0;
					for (auto svWithGamma : svVectorsWithGammasGlobalPool)
					{
						svWithOtherIds.emplace_back(svWithGamma.id, svWithGamma.classValue);
						++idCounter;
					}

					std::vector<DatasetVector> tempTR = previousUncertain;
					std::vector<uint64_t> tempTR_IDS = trIds;


					previousUncertain.insert(previousUncertain.end(), svWithOtherIds.begin(), svWithOtherIds.end());
					tempTR_IDS.insert(tempTR_IDS.end(), svIdsGlobal.begin(), svIdsGlobal.end());



					svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(previousUncertain) };
					auto tempTR2 = uncertainTraining.convertChromosome(m_joined_T_V);

					createVisualizationNewFlow(tempTR2, m_joined_T_V, m_id, svm, tempTR_IDS, validationIDS);

				}
				else
				{
					//createVisualization(tr_l, val_l, m_id, svm, trIds, validationIDS);

					if (m_full_config.getValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet"))
					{
						//USE FULL VALIDATION IN HERE
						createVisualizationNewFlow(fullSet, joinSets(fullSet, m_loadingWorkflow->getValidationSet()), m_id, svm, fullSetIds, fullSetIds);
					}
					else
					{
						createVisualizationNewFlow(fullSet, fullSet, m_id, svm, fullSetIds, fullSetIds);
					}
				}

				/*auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());


				std::ofstream tempFile(m_config.outputFolderPath.string() + "\\classDidivedAnswers_node=" + std::to_string(m_id) + ".txt");
				tempFile << "# width, heigh, positive, negative\n";


				for (int i = 0; i < 500 * 500; ++i)
				{
					int w = i / 500;
					int h = i % 500;

					std::vector<float> sample = { static_cast<float>(h) / static_cast<float>(500), static_cast<float>(w) / static_cast<float>(500) };

					auto [pos, neg, label] = res->classifyPositiveNegative(sample);

					tempFile << w << ", " << h << ", " << pos << ", " << neg << "\n";
				}

				tempFile.close();*/
			}

			auto [uncertainTrainingSet, uncertainIds] = getUncertainDataset(tr_l, trIds, svm);
			auto [uncertainValidationSet, uncertainIdsValidation] = getUncertainDataset(val_l, validationIDS, svm);
			previousUncertain = uncertainTrainingSet;

			auto [uncertainFullSet, uncertainIdsFullSet] = getUncertainDataset(fullSet, fullSetIds, svm);
			auto countFullSet = countClasses(uncertainFullSet);
			nothingImprovedInFullSet = fullSet.size() == uncertainFullSet.size();
			svmComponents::SvmTrainingSetChromosome uncertainFullSetCh{ std::move(uncertainFullSet) };
			fullSet = uncertainFullSetCh.convertChromosome(trainingSet);
			fullSetIds = uncertainIdsFullSet;


			auto count = countClasses(uncertainValidationSet);
			countTR = countClasses(uncertainTrainingSet);


			if (m_full_config.getValue<bool>("Svm.EnsembleTree.UseImbalanceRatio"))
			{
				auto min = std::min(countTR[0], countTR[1]);
				auto max = std::max(countTR[0], countTR[1]);
				auto imbalanceRatio = static_cast<float>(max) / static_cast<float>(min);
				if (imbalanceRatio > 5)
				{
					auto minorityClass = countTR[0] < countTR[1] ? 0 : 1;

					auto samples = tr_l.getSamples();
					auto labels = tr_l.getLabels();

					for (auto x = 0u; x < samples.size(); x++)
					{
						if (labels[x] == minorityClass)
						{
							uncertainTrainingSet.emplace_back(trIds[x], labels[x]);
							uncertainIds.emplace_back(trIds[x]);
						}
					}


					/*uncertainTrainingSet.insert(uncertainTrainingSet.end(), svVectorsGlobal.begin(), svVectorsGlobal.end());
					uncertainIds.insert(uncertainIds.end(), svIdsGlobal.begin(), svIdsGlobal.end());*/
				}
			}


			//if training or validation is empty we cannot proceed further
			done = std::all_of(count.begin(), count.end(), [&](uint64_t classes)
				{
					return classes == 0;
				});
			doneTR = std::any_of(countTR.begin(), countTR.end(), [&](uint64_t classes)
				{
					return classes == 0;
				});
			nothingImprovedInTraining = tr_l.size() == uncertainTrainingSet.size();
			nothingImprovedInValidation = val_l.size() == uncertainValidationSet.size();

			//SV region
			auto [svDatasetVectors, svIds] = getSVsIds(tr_l, trIds, svm);


			svIdsGlobal.insert(svIdsGlobal.end(), svIds.begin(), svIds.end());
			svVectorsGlobal.insert(svVectorsGlobal.end(), svDatasetVectors.begin(), svDatasetVectors.end());

			//save to pool with vectors
			if (m_useDasvmKernel && !linearIsBetter)
			{
				auto [svDatasetVectors2, svIds2] = getSVsIds(trainingSet, ids, svm);

				auto gammas = svm->getGammas();
				for (auto i = 0u; i < svDatasetVectors2.size(); ++i)
				{
					auto sv = svDatasetVectors2[i];
					if (svVectorsWithGammaIdSet.emplace(sv.id).second)
					{
						svVectorsWithGammasGlobalPool.emplace_back(sv.id, sv.classValue, gammas[i]);
					}
				}
			}

			svVectorsToPassDown.clear();
			svVectorsWithGammas.clear();

			//global
			if (m_algorithmConfig.m_SvMode == SvMode::all)
			{
				//global setting so taking all of the previous SVs into account
				int idCounter = 0;
				for (auto svWithGamma : svVectorsWithGammasGlobalPool)
				{
					if (m_useDasvmKernel && !linearIsBetter)
					{
						//svVectorsWithGammas.emplace_back(uncertainIds.size() + idCounter, svWithGamma.classValue, svWithGamma.gamma);
						svVectorsWithGammas.emplace_back(svWithGamma.id, svWithGamma.classValue, svWithGamma.gamma);
					}
					svVectorsToPassDown.emplace_back(uncertainIds.size() + idCounter, svWithGamma.classValue);
					++idCounter;
				}


				if (!m_useDasvmKernel)
				{
					uncertainTrainingSet.insert(uncertainTrainingSet.end(), svVectorsGlobal.begin(), svVectorsGlobal.end());
					uncertainIds.insert(uncertainIds.end(), svIdsGlobal.begin(), svIdsGlobal.end());
				}
			}
			else if (m_algorithmConfig.m_SvMode == SvMode::previousOnes) //only last one vectors
			{
				LOG_F(ERROR, "Getting only previous SV not implemented with new flow");
			}
			else if (m_algorithmConfig.m_SvMode == SvMode::none)
			{
				; //do nothing in this case
			}



			svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(uncertainTrainingSet) };
			tr_l = uncertainTraining.convertChromosome(trainingSet);
			svmComponents::SvmTrainingSetChromosome uncertainValidation{ std::move(uncertainValidationSet) };
			val_l = uncertainValidation.convertChromosome(trainingSet);

			trIds = uncertainIds;
			validationIDS = uncertainIdsValidation;


			if (m_debugLog)
			{
				auto [uncertainTestSet, uncertainIdsTest] = getUncertainDataset(test_l, testSetIds, svm);
				svmComponents::SvmTrainingSetChromosome uncertainTest{ std::move(uncertainTestSet) };
				test_l = uncertainTest.convertChromosome(m_loadingWorkflow->getTestSet());
				testSetIds = uncertainIdsTest;

				saveRegions(fullSetIds, fullSetIds, testSetIds, true);
			}

			m_listLength++;
			m_id++;
			if (stopCondition())
			{
				break;
			}

			try
			{
				if (!tr_l.empty() || !val_l.empty())
				{
				//RESAMPLE AGAIN FROM UNCERTAIN SETS, remember about ids
				auto result = resample(tr_l, trIds, val_l, validationIDS);

				tr_l = std::get<0>(result);
				trIds = std::get<1>(result);
				val_l = std::get<2>(result);
				validationIDS = std::get<3>(result);
				}
			}
			catch (const std::exception& e)
			{
				LOG_F(ERROR, "Error: %s", e.what());
				std::cout << e.what();
				throw;
			}
				
			countTR = countClasses(tr_l);
			if ((tr_l.size() < numberOfClasses * min_k ||
				std::any_of(countTR.begin(), countTR.end(), [&](uint64_t classes) {return classes < min_k; }))
				) //at least one regular node
			{
					datasetTooSmall = true; //one of the end conditions
					break;
			}

	
			last_svm = svm;
			temp->m_next = std::make_shared<phd::svm::ListNodeSvm>();
			temp = temp->m_next;
		}

		auto btos = [&](bool cond)
		{
			return cond ? "true" : "false";
		};

		LOG_F(INFO, "Stop conditions: nothingImprovedInFull_T+V_Set=%s,   datasetTooSmall=%s, max_length=%s", btos(nothingImprovedInFullSet), btos(datasetTooSmall), btos(m_id >= max_length));
		LOG_F(INFO, "Size of uncertain validation %d,  original size %d", val_l.size(), m_loadingWorkflow->getValidationSet().size());
		LOG_F(INFO, "Size of uncertain training %d,  original size %d", tr_l.size(), m_loadingWorkflow->getTraningSet().size());


		addLastNodeWithFullSvm(temp);


		return root_;
	}
	catch (const std::exception& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
		std::cout << e.what();
		throw;
	}

}



std::shared_ptr<phd::svm::ListNodeSvm> EnsembleTreeWorkflow::trainHelper(std::shared_ptr<phd::svm::ListNodeSvm>& root_,
	                                                                         const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                                         const std::vector<uint64_t>& ids,
	                                                                         const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                                                         const std::vector<uint64_t>& valIds)
{

	
	try
	{
		const int max_length = 50000;
		//const int max_length = 3;
		const int numberOfClasses = 2;
		const int min_k = m_algorithmConfig.m_trainingSetOptimization->getInitialTrainingSetSize();

		std::shared_ptr<phd::svm::ISvm> svm;
		std::shared_ptr<phd::svm::ISvm> last_svm;
		bool datasetTooSmall = false;
		bool nothingImprovedInValidation = false;
		bool nothingImprovedInTraining = false;
		bool done = false;
		bool doneTR = false;
		dataset::Dataset<std::vector<float>, float> tr_l = trainingSet;
		dataset::Dataset<std::vector<float>, float> val_l = validationSet;
		dataset::Dataset<std::vector<float>, float> test_l = m_loadingWorkflow->getTestSet();
		std::vector<uint64_t> trIds = ids;
		std::vector<uint64_t> validationIDS = valIds;

		std::vector<uint64_t> svIdsGlobal;
		std::vector<DatasetVector> svVectorsGlobal;
		std::vector<DatasetVector> svVectorsToPassDown;

		std::vector<DatasetVector> previousUncertain;

		std::vector<Gene> svVectorsWithGammas;
		std::vector<Gene> svVectorsWithGammasGlobalPool;
		std::unordered_set<uint64_t> svVectorsWithGammaIdSet;

		std::vector<uint64_t> testSetIds;
		for (auto i = 0u; i < m_loadingWorkflow->getTestSet().size(); ++i)
		{
			testSetIds.emplace_back(i);
		}

		std::vector<uint64_t> countTR = countClasses(trainingSet);

		auto temp = root_;
		auto stopCondition = [&]()
		{
			return (nothingImprovedInValidation || nothingImprovedInTraining || m_id >= max_length || datasetTooSmall || done || doneTR);
			//return (nothingImprovedInTraining || m_id >= max_length || datasetTooSmall || done || doneTR);
		};

		while (!stopCondition()) //limit size of ensemble for visualization speed problems
		{
			DatasetLoaderHelper datasets(tr_l, val_l, m_loadingWorkflow->getTestSet());

			if (m_listLength >= 1 && (tr_l.size() < numberOfClasses * min_k || val_l.size() < numberOfClasses * min_k ||
				std::any_of(countTR.begin(), countTR.end(), [&](uint64_t classes) {return classes < min_k;}))
				) //at least one regular node
			{
				DatasetLoaderHelper datasets2(tr_l, m_loadingWorkflow->getValidationSet(), m_loadingWorkflow->getTestSet());
				////DatasetLoaderHelper datasets2(m_loadingWorkflow->getTraningSet(), m_loadingWorkflow->getValidationSet(), m_loadingWorkflow->getTestSet());
				////train full SVM with leftovers
				svm = runGridSearch(datasets2, last_svm);

				temp->m_svm = svm;
				datasetTooSmall = true; //one of the end conditions
				break;
			}
			else
			{
				std::cout << "ID:" << m_listLength << "\n";
				auto configurationForNode = EnsembleTreeWorkflowConfig(m_full_config, datasets);

				SvmHelper helper(m_config, configurationForNode, datasets, m_algorithmConfig.m_addSvToTraining,
				                 svVectorsToPassDown, *m_loadingWorkflow, svVectorsWithGammas,
				                 m_useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config);
				svm = helper.run();
				temp->m_svm = svm;
				
			}

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				if(m_useDasvmKernel && !previousUncertain.empty())
				{
					std::vector<DatasetVector> svWithOtherIds;
					
						int idCounter = 0;
						for (auto svWithGamma : svVectorsWithGammasGlobalPool)
						{
							svWithOtherIds.emplace_back(svWithGamma.id, svWithGamma.classValue);
							++idCounter;
						}

						std::vector<DatasetVector> tempTR = previousUncertain;
						std::vector<uint64_t> tempTR_IDS = trIds;

						
						previousUncertain.insert(previousUncertain.end(), svWithOtherIds.begin(), svWithOtherIds.end());
						tempTR_IDS.insert(tempTR_IDS.end(), svIdsGlobal.begin(), svIdsGlobal.end());
					
					

					svmComponents::SvmTrainingSetChromosome uncertainTraining{ std::move(previousUncertain) };
					auto tempTR2 = uncertainTraining.convertChromosome(m_loadingWorkflow->getTraningSet());

					createVisualization(tempTR2, val_l, m_id, svm, tempTR_IDS, validationIDS);
					
				}
				else
				{
					//createVisualization(tr_l, val_l, m_id, svm, trIds, validationIDS);
				}
			}

			auto [uncertainTrainingSet, uncertainIds] = getUncertainDataset(tr_l, trIds, svm);
			auto [uncertainValidationSet, uncertainIdsValidation] = getUncertainDataset(val_l, validationIDS, svm);
			previousUncertain = uncertainTrainingSet;



			
			auto count = countClasses(uncertainValidationSet);
			countTR = countClasses(uncertainTrainingSet);


			if (m_full_config.getValue<bool>("Svm.EnsembleTree.UseImbalanceRatio"))
			{
				auto min = std::min(countTR[0], countTR[1]);
				auto max = std::max(countTR[0], countTR[1]);
				auto imbalanceRatio = static_cast<float>(max) / static_cast<float>(min);
				if(imbalanceRatio > 5)
				{
					auto minorityClass = countTR[0] < countTR[1] ? 0 : 1;

					auto samples = tr_l.getSamples();
					auto labels = tr_l.getLabels();

					for(auto x = 0u; x < samples.size(); x++)
					{
						if(labels[x] == minorityClass)
						{
							uncertainTrainingSet.emplace_back(trIds[x], labels[x]);
							uncertainIds.emplace_back(trIds[x]);
						}
					}
					

					/*uncertainTrainingSet.insert(uncertainTrainingSet.end(), svVectorsGlobal.begin(), svVectorsGlobal.end());
					uncertainIds.insert(uncertainIds.end(), svIdsGlobal.begin(), svIdsGlobal.end());*/
				}
			}
			

			//if training or validation is empty we cannot proceed further
			done = std::all_of(count.begin(), count.end(), [&](uint64_t classes)
			{
				return classes == 0;
			});
			doneTR = std::any_of(countTR.begin(), countTR.end(), [&](uint64_t classes)
			{
				return classes == 0;
			});
			nothingImprovedInTraining = tr_l.size() == uncertainTrainingSet.size();
			nothingImprovedInValidation = val_l.size() == uncertainValidationSet.size();

			//SV region
			auto [svDatasetVectors, svIds] = getSVsIds(tr_l, trIds, svm);
			

			svIdsGlobal.insert(svIdsGlobal.end(), svIds.begin(), svIds.end());
			svVectorsGlobal.insert(svVectorsGlobal.end(), svDatasetVectors.begin(), svDatasetVectors.end());

			//save to pool with vectors
			if (m_useDasvmKernel)
			{
				auto [svDatasetVectors2, svIds2] = getSVsIds(trainingSet, ids, svm);
				
				auto gammas = svm->getGammas();
				for(auto i = 0u; i< svDatasetVectors2.size(); ++i)
				{
					auto sv = svDatasetVectors2[i];
					if(svVectorsWithGammaIdSet.emplace(sv.id).second)
					{
						svVectorsWithGammasGlobalPool.emplace_back(sv.id, sv.classValue, gammas[i]);
					}
				}
			}

			svVectorsToPassDown.clear();
			svVectorsWithGammas.clear();

			//global
			if (m_algorithmConfig.m_SvMode == SvMode::all)
			{
				//global setting so taking all of the previous SVs into account
				int idCounter = 0;
				for (auto svWithGamma : svVectorsWithGammasGlobalPool)
				{
					if (m_useDasvmKernel)
					{
						//svVectorsWithGammas.emplace_back(uncertainIds.size() + idCounter, svWithGamma.classValue, svWithGamma.gamma);
						svVectorsWithGammas.emplace_back(svWithGamma.id, svWithGamma.classValue, svWithGamma.gamma);
					}
					svVectorsToPassDown.emplace_back(uncertainIds.size() + idCounter, svWithGamma.classValue);
					++idCounter;
				}


				if( !m_useDasvmKernel)
				{
					uncertainTrainingSet.insert(uncertainTrainingSet.end(), svVectorsGlobal.begin(), svVectorsGlobal.end());
					uncertainIds.insert(uncertainIds.end(), svIdsGlobal.begin(), svIdsGlobal.end());
				}
			}
			else if (m_algorithmConfig.m_SvMode == SvMode::previousOnes) //only last one vectors
			{

				////use only sv vectors of previous node in here, gamma from previous svm
				//int idCounter = 0;
				//for (auto sv : svDatasetVectors)
				//{
				//	if (m_useDasvmKernel)
				//	{
				//		svVectorsWithGammas.emplace_back(uncertainIds.size() + idCounter, sv.classValue, svm->getGamma());
				//	}
				//	svVectorsToPassDown.emplace_back(uncertainIds.size() + idCounter, sv.classValue);
				//	++idCounter;
				//}

				//if (!m_useDasvmKernel) //This vectors must not be added 
				//{
				//	uncertainTrainingSet.insert(uncertainTrainingSet.end(), svDatasetVectors.begin(), svDatasetVectors.end());
				//	uncertainIds.insert(uncertainIds.end(), svIds.begin(), svIds.end());
				//}
			}
			else if (m_algorithmConfig.m_SvMode == SvMode::none)
			{
				; //do nothing in this case
			}

		
			
			svmComponents::SvmTrainingSetChromosome uncertainTraining{std::move(uncertainTrainingSet)};
			tr_l = uncertainTraining.convertChromosome(m_loadingWorkflow->getTraningSet());
			svmComponents::SvmTrainingSetChromosome uncertainValidation{std::move(uncertainValidationSet)};
			val_l = uncertainValidation.convertChromosome(m_loadingWorkflow->getValidationSet());
			
			trIds = uncertainIds;
			validationIDS = uncertainIdsValidation;
			

			if(m_debugLog)
			{
				auto [uncertainTestSet, uncertainIdsTest] = getUncertainDataset(test_l, testSetIds, svm);
				svmComponents::SvmTrainingSetChromosome uncertainTest{ std::move(uncertainTestSet) };
				test_l = uncertainTest.convertChromosome(m_loadingWorkflow->getTestSet());
				testSetIds = uncertainIdsTest;
				
				saveRegions(trIds, validationIDS, testSetIds);
			}

			if (m_listLength >= 1 && (tr_l.size() < numberOfClasses * min_k || val_l.size() < numberOfClasses * min_k ||
				std::any_of(countTR.begin(), countTR.end(), [&](uint64_t classes) {return classes < min_k; }))
				) //at least one regular node
			{
				datasetTooSmall = true; //one of the end conditions
			}
			
			if (stopCondition())
			{
				break;
			}

			m_listLength++;
			m_id++;
			last_svm = svm;
			temp->m_next = std::make_shared<phd::svm::ListNodeSvm>();
			temp = temp->m_next;
		}

		auto btos = [&](bool cond)
		{
			return cond ? "true" : "false";
		};

		LOG_F(INFO, "Stop conditions: nothingImprovedInValidation=%s,  nothingImprovedInTraining=%s,  datasetTooSmal=%s,  done=%s,  doneTR=%s",
			btos(nothingImprovedInValidation), btos(nothingImprovedInTraining), btos(datasetTooSmall), btos(done), btos(doneTR));
		LOG_F(INFO, "Size of uncertain validation %d,  original size %d", val_l.size(), m_loadingWorkflow->getValidationSet().size());
		LOG_F(INFO, "Size of uncertain training %d,  original size %d", tr_l.size(), m_loadingWorkflow->getTraningSet().size());


		addLastNodeWithFullSvm(temp);

		
		return root_;
	}
	catch (const std::exception& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
		std::cout << e.what();
		throw;
	}

}


void EnsembleTreeWorkflow::train(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	auto newSampling = m_full_config.getValue<bool>("Svm.EnsembleTree.NewDatasetSampling");

	if (newSampling)
	{
		//join train and validation in here
		
		
		std::call_once(m_initValidationSize, [&]()
			{
				auto classCount = svmUtils::countLabels(2, m_joined_T_V);

				if (std::all_of(classCount.begin(), classCount.end(), [](auto value)
					{
						return value > 0;
					}))
				{
					constexpr auto max_val = 5000.0;

					auto percent = max_val / static_cast<double>(classCount[0] + classCount[1]);
					if (percent > 0.25)
					{
						percent = 0.25;
					}

					
					m_validationNegative = static_cast<unsigned>(std::round(classCount[0] * percent));
					m_validationPositive = static_cast<unsigned>(std::round(classCount[1] * percent));
				}
				else
				{
					throw std::exception("Only single class provided in dataset");
				}
			});
		

		std::vector<uint64_t> set;
		for (auto i = 0u; i < m_joined_T_V.size(); ++i)
		{
			set.emplace_back(i);
		}
		std::vector<uint64_t> vset;

		dataset::Dataset<std::vector<float>, float> empty;
		auto temp = std::make_shared<phd::svm::ListNodeSvm>(nullptr);
		root = trainHelperNewDatasetFlow(temp, m_joined_T_V, set, empty, vset);
		
	}
	else
	{
		std::vector<uint64_t> set;
		for (auto i = 0u; i < trainingSet.size(); ++i)
		{
			set.emplace_back(i);
		}
		std::vector<uint64_t> vset;
		for (auto i = 0u; i < m_loadingWorkflow->getValidationSet().size(); ++i)
		{
			vset.emplace_back(i);
		}

		auto temp = std::make_shared<phd::svm::ListNodeSvm>(nullptr);
		root = trainHelper(temp, trainingSet, set, m_loadingWorkflow->getValidationSet(), vset);
	}
}


void saveDataset(const dataset::Dataset<std::vector<float>, float>& set, const std::vector<unsigned long long>& ids, std::string name)
{
	std::ofstream output(name);

	for(auto i = 0u; i < ids.size(); ++i)
	{
		auto sample = set.getSample(ids[i]);
		for(auto& feature : sample)
		{
			output << feature << ",";
		}
		output << set.getLabel(ids[i]) << "\n";
	}
	output.close();
}

std::vector<unsigned long long> getCertainIdsFromUncertain(const dataset::Dataset<std::vector<float>, float>& set, const std::vector<unsigned long long>& ids)
{
	
	std::vector<unsigned long long> certainSet;
	for (auto i = 0u; i < set.size(); ++i)
	{
		if (std::find(ids.begin(), ids.end(), i) == ids.end())
		{
			certainSet.emplace_back(i);
		}
	}
	return certainSet;
}

void EnsembleTreeWorkflow::saveRegions(const std::vector<unsigned long long>& trainUncertainIds,
                                       const std::vector<unsigned long long>& validationIds,
                                       const std::vector<unsigned long long>& testSetIds,
									   bool useTrainedWithValidation) const
{
	dataset::Dataset<std::vector<float>, float> train = m_loadingWorkflow->getTraningSet();
	dataset::Dataset<std::vector<float>, float> val = m_loadingWorkflow->getValidationSet();
	dataset::Dataset<std::vector<float>, float> test = m_loadingWorkflow->getTestSet();
	
	if(useTrainedWithValidation)
	{
		train = val = m_joined_T_V;
	}

	auto certainTR = getCertainIdsFromUncertain(train, trainUncertainIds);
	auto certainVAL = getCertainIdsFromUncertain(val, validationIds);
	auto certainTEST = getCertainIdsFromUncertain(test, testSetIds);


	auto outputBase = m_config.outputFolderPath.string() + "\\regions";
	//filesystem::FileSystem fs;
	std::filesystem::create_directories(outputBase);

	if (certainTR.size() + trainUncertainIds.size() != train.size()) 
	{ LOG_F(ERROR, "train set is not correct"); }
	if (certainVAL.size() + validationIds.size() != val.size()) 
	{ LOG_F(ERROR, "validation set is not correct"); }
	if (certainTEST.size() + testSetIds.size() != test.size())
	{ LOG_F(ERROR, "test set is not correct"); }

	saveDataset(train, trainUncertainIds, outputBase + "\\uncertainTrain_" + std::to_string(m_listLength) + ".csv");
	saveDataset(train, certainTR, outputBase + "\\CertainTrain_" + std::to_string(m_listLength) + ".csv");

	saveDataset(val, validationIds, outputBase + "\\uncertainVAL_" + std::to_string(m_listLength) + ".csv");
	saveDataset(val, certainVAL, outputBase + "\\CertainVAL_" + std::to_string(m_listLength) + ".csv");

	saveDataset(test, testSetIds, outputBase + "\\uncertainTEST_" + std::to_string(m_listLength) + ".csv");
	saveDataset(test, certainTEST, outputBase + "\\CertainTEST_" + std::to_string(m_listLength) + ".csv");
}


dataset::Dataset<std::vector<float>, float> EnsembleTreeWorkflow::getCertain(std::shared_ptr<phd::svm::ISvm> svm,
                                                                             const dataset::Dataset<std::vector<float>, float>& set, bool certain)
{
	std::vector<uint64_t> vset;
	for (auto i = 0u; i < set.size(); ++i)
	{
		vset.emplace_back(i);
	}

	if (certain)
	{
		auto [vectors, ids] = getCertainDataset(set, vset, svm);
		svmComponents::SvmTrainingSetChromosome uncertainTraining{std::move(vectors)};
		auto validationCertainDataset = uncertainTraining.convertChromosome(set);

		return validationCertainDataset;
	}
	else
	{
		auto [vectors, ids] = getUncertainDataset(set, vset, svm);
		svmComponents::SvmTrainingSetChromosome uncertainTraining{std::move(vectors)};
		auto validationCertainDataset = uncertainTraining.convertChromosome(set);

		return validationCertainDataset;
	}
}

void EnsembleTreeWorkflow::chartDataSave(std::shared_ptr<phd::svm::libSvmImplementation> svm, int list_length,
                                         const dataset::Dataset<std::vector<float>, float>& val,
                                         const dataset::Dataset<std::vector<float>, float>& test)
{
	auto separator = ';';
	auto path = m_config.outputFolderPath.string() + "//nodeChartData__" + std::to_string(list_length) + ".txt";
	std::ofstream data(path);
	data << svm->getNegativeNormalizedCertainty() << separator << svm->getPositiveNormalizedCertainty() << "\n";

	auto samples = val.getSamples();
	auto targets = val.getLabels();

	std::vector<std::pair<float, int>> results;
	results.reserve(targets.size());

	for (auto i = 0u; i < targets.size(); i++)
	{
		auto temp = std::make_pair(static_cast<float>(svm->classifyHyperplaneDistance(samples[i])),
		                           static_cast<int>(targets[i]));

		results.emplace_back(temp);
	}

	std::sort(results.begin(), results.end(), [&](const auto& a, const auto& b)
	{
		return std::get<0>(a) < std::get<0>(b);
	});
	auto min = results[0].first;
	auto max = results[results.size() - 1].first;

	for (auto& pair : results)
	{
		if (pair.first < 0)
			pair.first = - (pair.first / min);
		else
			pair.first = pair.first / max;
		data << pair.first << separator << pair.second << "\n";
	}

	auto negativeThr = 0.0f;
	if (svm->getNegativeCertainty() > 0)
		negativeThr = static_cast<float>(svm->getNegativeCertainty()) / max;
	else
		negativeThr = static_cast<float>(svm->getNegativeCertainty()) / min;

	data.close();

	//TEST SET FROM HERE
	auto path2 = m_config.outputFolderPath.string() + "//nodeChartDataTest__" + std::to_string(list_length) + ".txt";
	std::ofstream data2(path2);
	samples = test.getSamples();
	targets = test.getLabels();
	std::vector<std::pair<float, int>> results2;
	results2.reserve(targets.size());

	for (auto i = 0u; i < targets.size(); i++)
	{
		auto temp = std::make_pair(static_cast<float>(svm->classifyHyperplaneDistance(samples[i])),
		                           static_cast<int>(targets[i]));

		results2.emplace_back(temp);
	}

	std::sort(results2.begin(), results2.end(), [&](const auto& a, const auto& b)
	{
		return std::get<0>(a) < std::get<0>(b);
	});

	for (auto& pair : results2)
	{
		if (pair.first < 0)
			pair.first = -(pair.first / min);
		else
			pair.first = pair.first / max;
		data2 << pair.first << separator << pair.second << "\n";
	}
	data2.close();
}

void EnsembleTreeWorkflow::perNodeVisualization(std::shared_ptr<phd::svm::EnsembleListSvm> /*ensemble*/, int /*length*/)
{
	svmComponents::SvmVisualization visualization_per_node;
	std::filesystem::path m_pngNameSource;
	//auto image = visualization_per_node.createEnsembleVisualizationPerNode(*ensemble, 500, 500, m_loadingWorkflow->getTraningSet(),
	//                                                                       m_loadingWorkflow->getValidationSet(), m_loadingWorkflow->getTestSet());

	//SvmWokrflowConfiguration config_copy3{ "", "", "", m_config.outputFolderPath, "ExtraTreeTest" + std::to_string(0), "" };
	//setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
	//m_savePngElement.launch(image, m_pngNameSource);
	
	/*auto image = visualization_per_node.createEnsembleVisualizationPerNode(*ensemble, 500, 500, m_joined_T_V,
		m_joined_T_V, m_loadingWorkflow->getTestSet());
	SvmWokrflowConfiguration config_copy3{"", "", "", m_config.outputFolderPath, "NewClassificationScheme__" + std::to_string(length), ""};
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
	m_savePngElement.launch(image, m_pngNameSource);*/
}



std::pair<dataset::Dataset<std::vector<float>, float>, dataset::Dataset<std::vector<float>, float>>
EnsembleTreeWorkflow::scoreEnsemble(std::shared_ptr<phd::svm::EnsembleListSvm> ensemble)
{
	svmComponents::BaseSvmChromosome to_test2;
	to_test2.updateClassifier(ensemble);
	std::vector<svmComponents::BaseSvmChromosome> vec2;
	vec2.emplace_back(to_test2);

	geneticComponents::Population<svmComponents::BaseSvmChromosome> pop2{std::move(vec2)};

	m_validationTest.launch(pop2, m_loadingWorkflow->getValidationSet());
	auto validationCM = pop2.getBestOne().getConfusionMatrix();
	auto validationFitness = pop2.getBestOne().getFitness();
	auto SvNumber = pop2.getBestOne().getNumberOfSupportVectors();

	m_validationTest.launch(pop2, m_loadingWorkflow->getTestSet());
	auto testCM = pop2.getBestOne().getConfusionMatrix();
	auto testFitness = pop2.getBestOne().getFitness();
	
	auto validationCertain = getCertain(ensemble, m_loadingWorkflow->getValidationSet(), true);

	m_validationTest.launch(pop2, validationCertain);
	auto validationCertainFitness = pop2.getBestOne().getFitness();
	auto validationCertainCM = pop2.getBestOne().getConfusionMatrix();

	auto testCertain = getCertain(ensemble, m_loadingWorkflow->getTestSet(), true);

	m_validationTest.launch(pop2, testCertain);
	auto testCertainFitness = pop2.getBestOne().getFitness();
	auto testCertainCM = pop2.getBestOne().getConfusionMatrix();

	auto validationUncertain = getCertain(ensemble, m_loadingWorkflow->getValidationSet(), false);

	m_validationTest.launch(pop2, validationUncertain);
	auto validationUncertainFitness = pop2.getBestOne().getFitness();
	auto validationUncertainCM = pop2.getBestOne().getConfusionMatrix();

	auto testUncertain = getCertain(ensemble, m_loadingWorkflow->getTestSet(), false);

	m_validationTest.launch(pop2, testUncertain);
	auto testUncertainFitness = pop2.getBestOne().getFitness();
	auto testUncertainCM = pop2.getBestOne().getConfusionMatrix();

	try
	{
		auto sep = '\t';
		auto path = m_config.outputFolderPath.string() + "//ensembleLog.txt";
		std::ofstream ensembleFile(path, std::ios_base::app);
		if (is_empty(path))
		{
			ensembleFile << "### length\tSV \tAuc V\tAuc Test\tConfusion matrix validation(4 numbers)\tConfusion matrix test(4 numbers)\n";
		}
		ensembleFile << ensemble->list_length << sep << SvNumber << sep << validationFitness << sep << testFitness << sep << validationCM << sep << testCM <<
				sep;
		ensembleFile << validationCertainFitness << sep << validationCertainCM << sep << testCertainFitness << sep << testCertainCM << sep;
		ensembleFile << validationUncertainFitness << sep << validationUncertainCM << sep << testUncertainFitness << sep << testUncertainCM << "\n";

		ensembleFile.close();
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, exception.what());
		std::cout << exception.what();
		throw;
	}

	return std::make_pair(validationUncertain, testUncertain);
}

std::shared_ptr<phd::svm::ISvm> EnsembleTreeWorkflow::runGridSearch(IDatasetLoader& workflow, std::shared_ptr<phd::svm::ISvm>& /*previousNode*/)
{
	std::shared_ptr<svmComponents::BaseKernelGridSearch> kernel;

	//if (previousNode->getKernelType() == phd::svm::KernelTypes::Rbf)
	//{
	//	kernel = std::make_shared<svmComponents::RbfKernel>(cv::ml::ParamGrid(0.001, 1050, 10), cv::ml::ParamGrid(0.001, 1050, 10), false);
	//}
	//else if (previousNode->getKernelType() == phd::svm::KernelTypes::Linear)
	//{
	//	kernel = std::make_shared<svmComponents::LinearKernel>(cv::ml::ParamGrid(0.001, 1050, 10), false);
	//}
	//else
	{
		kernel = std::make_shared<svmComponents::RbfKernel>(cv::ml::ParamGrid(0.001, 1050, 10), cv::ml::ParamGrid(0.001, 1050, 10), false);
	}

	//ORIGINAL VERSION
	GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	                                                                       1,
	                                                                       1,
	                                                                       0, //subset size = 0 means full training set
	                                                                       1,
	                                                                       std::make_shared<svmComponents::SvmKernelTraining>(m_algorithmConfig.m_svmConfig,
	                                                                                                                          m_algorithmConfig
	                                                                                                                          .m_svmConfig.m_estimationType ==
	                                                                                                                          svmComponents::svmMetricType::
	                                                                                                                          CertainAccuracy),
	                                                                       kernel),
	                      workflow);
	
	
	//GridSearchWorkflow gs(m_config, svmComponents::GridSearchConfiguration(m_algorithmConfig.m_svmConfig,
	//	1,
	//	1,
	//	32, //subset size = 0 means full training set
	//	1,
	//	std::make_shared<svmComponents::SvmKernelTraining>(m_algorithmConfig.m_svmConfig,
	//		m_algorithmConfig
	//		.m_svmConfig.m_estimationType ==
	//		svmComponents::svmMetricType::
	//		CertainAccuracy),
	//	kernel),
	//	workflow);

	auto bestOne = gs.run();

	return bestOne;
}






void EnsembleTreeWorkflow::update()
{

	//load classifier
	auto trained_one = phd::svm::EnsembleListSvm(R"(D:\ENSEMBLE_825_GECCO_2D_update\base\1\EnsembleList_ET_NF_Class_Bias__2022-01-18-23_11_35.452_\2022-01-18-23_11_37.150__EnsembleList_ET_NF_Class_Bias1_0fold_1_svmModel.xml)", true);

		
	root = trained_one.root;
		
	//proper datasets, select only uncertain part
	m_updateTr = processUncertain(m_loadingWorkflow->getTraningSet(), trained_one);
	m_updateVal = processUncertain(m_loadingWorkflow->getValidationSet(), trained_one);
	m_updateTest = m_loadingWorkflow->getTestSet();
		
	//regular training from here
	DatasetLoaderHelper *newDataset = new DatasetLoaderHelper(m_updateTr, m_updateVal, m_updateTest);

	
	m_loadingWorkflow = (newDataset);
	m_joined_T_V = joinSets(m_loadingWorkflow->getTraningSet(), m_loadingWorkflow->getValidationSet());
	//visualizations
		
}


std::shared_ptr<phd::svm::ISvm> EnsembleTreeWorkflow::run()
{
	try
	{
		//update example
		//update();
		
		train(m_loadingWorkflow->getTraningSet());
		//Used for ExtraTree visualization 
		//std::shared_ptr<phd::svm::ISvm> m_svm = std::make_shared<phd::svm::libSvmImplementation>();
		//m_svm->train(m_loadingWorkflow->getTraningSet());
		//root = std::make_shared<phd::svm::ListNodeSvm>(m_svm);

		//update example
		/*auto trained_one = phd::svm::EnsembleListSvm(R"(D:\ENSEMBLE_825_GECCO_2D_update\base\1\EnsembleList_ET_NF_Class_Bias__2022-01-18-23_11_35.452_\2022-01-18-23_11_37.150__EnsembleList_ET_NF_Class_Bias1_0fold_1_svmModel.xml)", true);
		auto tempNode = root;
		root = trained_one.root;
		root->m_next = tempNode;*/
		
		std::shared_ptr<phd::svm::EnsembleListSvm> tree = std::make_shared<phd::svm::EnsembleListSvm>(root, m_listLength, m_newClassificationScheme);

		/*LOG_F(INFO, "Starting training ExtraTree");
		ExtraTreeWrapper extraTree;
		extraTree.train(m_joined_T_V);
		tree->m_treeEndNode = std::make_shared<ExtraTreeWrapper>(extraTree);
		LOG_F(INFO, "Done ExtraTree");*/
		
		BaseSvmChromosome to_test;
		to_test.updateClassifier(tree);
		std::vector<BaseSvmChromosome> vec;
		vec.emplace_back(to_test);
		Population<BaseSvmChromosome> pop{std::move(vec)};

		//Calculate AUC for end report
		m_validation.launch(pop, m_loadingWorkflow->getValidationSet());
		auto copy = pop;
		m_validationTest.launch(copy, m_loadingWorkflow->getTestSet());

		auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
		auto bestOneTestMatrix = copy[pop.getBestIndividualIndex()].getConfusionMatrix().value();
		auto featureNumber = m_loadingWorkflow->getValidationSet().getSamples()[0].size();

		m_resultLogger.createLogEntry(pop,
		                              copy,
		                              m_timer,
		                              m_algorithmName,
		                              0,
		                              svmComponents::Accuracy(bestOneConfustionMatrix),
		                              featureNumber,
		                              0,
		                              bestOneConfustionMatrix,
		                              bestOneTestMatrix);

		m_resultLogger.logToFile(m_resultFilePath);


		//TODO think about stopping timer at this point, everythin below is just some additional stuff calculating
		
		auto list = tree->root;

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
		//tempEnsembleList0->m_treeEndNode = tree->m_treeEndNode;
		
		//TODO add visualization of ensmeble with single SVM
		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			perNodeVisualization(tempEnsembleList0, 0);
		}

		if(m_debugLog)
		{
			scoreEnsemble(tempEnsembleList0);
			chartDataSave(svmCopy1, length, m_loadingWorkflow->getValidationSet(), m_loadingWorkflow->getTestSet());
		}
			
		auto uncertainVal = getCertain(svmCopy1, m_loadingWorkflow->getValidationSet(), false);
		auto uncertainTest = getCertain(svmCopy1, m_loadingWorkflow->getTestSet(), false);

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
			//if (list->m_next == nullptr)
			//{
			//	tempEnsembleList->m_treeEndNode = tree->m_treeEndNode;
			//}
			auto svm_no_last_node = std::make_shared<phd::svm::EnsembleListSvm>(listWithoutLastNode, length - 1, m_newClassificationScheme);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				perNodeVisualization(tempEnsembleList, length);
			}

			if (m_debugLog)
			{
				scoreEnsemble(tempEnsembleList);

				if (!uncertainTest.empty() && !uncertainVal.empty())
				{
					chartDataSave(svmCopy, length, uncertainVal, uncertainTest);
				}
				uncertainVal = getCertain(svmCopy, uncertainVal, false);
				uncertainTest = getCertain(svmCopy, uncertainTest, false);
			}
			

			svmComponents::BaseSvmChromosome to_test2;
			to_test2.updateClassifier(tempEnsembleList);

			std::vector<svmComponents::BaseSvmChromosome> vec2;
			vec2.emplace_back(to_test2);
			geneticComponents::Population<svmComponents::BaseSvmChromosome> pop2{std::move(vec2)};
			svmComponents::SvmVisualization visualization3;
			std::filesystem::path m_pngNameSource;

			m_validation.launch(pop2, m_loadingWorkflow->getValidationSet());

			if (m_algorithmConfig.m_svmConfig.m_doVisualization) // only for debugging
			{
				auto tr = m_loadingWorkflow->getTraningSet();
				auto val = m_loadingWorkflow->getValidationSet();
				auto test = m_loadingWorkflow->getTestSet();
				
				if (list->m_next == nullptr)
				{
					auto image3 = visualization3.createEnsembleVisualization(*pop2.getBestOne().getClassifier(), 500, 500, tr, m_joined_T_V, test);
					SvmWokrflowConfiguration config_copy3{"", "", "", m_config.outputFolderPath, "ListEnsemble_All", ""};
					setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
					m_savePngElement.launch(image3, m_pngNameSource);
				}

				auto image4 = visualization3.createEnsembleImprobementVisualization(*pop2.getBestOne().getClassifier(), 500, 500,
				                                                                    tr, val, test,
				                                                                    tempEnsembleList->getSupportVectorsOfLastNode(), *svm_no_last_node);
				SvmWokrflowConfiguration config_copy3{"", "", "", m_config.outputFolderPath, "ListEnsemble_0-" + std::to_string(length), ""};
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
				m_savePngElement.launch(image4, m_pngNameSource, true);
			}

			auto anotherList = std::make_shared<phd::svm::ListNodeSvm>(nullptr, svmCopy);
			listWithoutLastNodeCreation->m_next = anotherList;
			listWithoutLastNodeCreation = listWithoutLastNodeCreation->m_next;

			list = list->m_next;
		}

		return tree;
	}
	catch (const std::exception& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
		std::cout << e.what() << "\n";
		throw;
	}
}

	
} // namespace genetic
