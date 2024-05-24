#include "libPlatform/loguru.hpp"
#include "libPlatform/StringUtils.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "BigSetsEnsemble.h"


#include "BigSetsSvmHelper.h"
#include "DefaultWorkflowConfigs.h"
#include "EnsembleUtils.h"
#include "GridSearchWorkflow.h"
#include "libRandom/MersenneTwister64Rng.h"
#include "SvmLib/libSvmImplementation.h"
#include "SvmLib/EnsembleListSvm.h"
#include "libSvmComponents/LinearKernel.h"
#include "libSvmComponents/RbfKernel.h"
#include "libGeneticSvm/DatasetLoaderHelper.h"
#include "SvmLib/VotingEnsemble.h"
#include "libSvmComponents/SvmAccuracyMetric.h"
#include "libSvmComponents/SvmAucMetric.h"
#include "SvmAlgorithmFactory.h"
#include "SvmEnsembleHelper.h"

namespace genetic
{
BigSetsEnsemble::BigSetsEnsemble(const SvmWokrflowConfiguration& config,
                                 EnsembleTreeWorkflowConfig algorithmConfig,
                                 IDatasetLoader& workflow,
                                 platform::Subtree full_config)
	: m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
	, m_algorithmConfig(algorithmConfig)
	, m_config(config)
	, m_loadingWorkflow(workflow)
	, m_full_config(full_config)
	, m_metric(std::make_unique<SvmAucMetric>(true)) //true for parallel classification
	, m_validation(*m_metric, false)
	, m_validationTest(*m_metric, true)
	, m_listLength(0)
	, m_id(0)
	, m_newClassificationScheme(true) //TODO extract to config
	, m_useDasvmKernel(algorithmConfig.m_useDasvmKernel)
	, m_debugLog(full_config.getValue<bool>("Svm.EnsembleTree.DebugLog"))
	, m_useFeatureSelection(full_config.getValue<bool>("Svm.EnsembleTree.UseFeatureSelction"))
	, m_numberOfLinearSVM(0)
	, m_numberOfRbfSVM(0)
{
	m_joined_T_V = joinSets(m_loadingWorkflow.getTraningSet(), m_loadingWorkflow.getValidationSet());
}

std::shared_ptr<phd::svm::ISvm> BigSetsEnsemble::run()
{
	try
	{
		train(m_loadingWorkflow.getTraningSet());
		
		//std::shared_ptr<phd::svm::EnsembleListSvm> tree = std::make_shared<phd::svm::EnsembleListSvm>(root, m_listLength, m_newClassificationScheme);

		/*LOG_F(INFO, "Starting training ExtraTree");
		ExtraTreeWrapper extraTree;
		extraTree.train(m_joined_T_V);
		tree->m_treeEndNode = std::make_shared<ExtraTreeWrapper>(extraTree);
		LOG_F(INFO, "Done ExtraTree");*/


		std::shared_ptr<phd::svm::VotingEnsemble> ensemble = std::make_shared<phd::svm::VotingEnsemble>(m_allEnsembles, m_weights);
		
		if(m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			for (auto i = 0u; i < m_allEnsembles.size(); ++i)
			{

				svmComponents::SvmVisualization visualization3;
				std::filesystem::path m_pngNameSource;
				auto cascade = m_allEnsembles[i];
				auto img = visualization3.createEnsembleVisualizationCertaintyMap(*cascade, 500, 500);
				SvmWokrflowConfiguration config_copy3{ "", "", "", m_config.outputFolderPath, "CERTAINTY_MAP_" + std::to_string(i), "" };
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
				m_savePngElement.launch(img, m_pngNameSource);
			}



			svmComponents::SvmVisualization visualization3;
			std::filesystem::path m_pngNameSource;
			int i = 0;
			for(auto cl : m_allEnsembles)
			{
				auto image3 = visualization3.createEnsembleVisualization(*cl, 500, 500, m_loadingWorkflow.getTraningSet(), m_joined_T_V, m_loadingWorkflow.getTestSet());
				SvmWokrflowConfiguration config_copy3{ "", "", "", m_config.outputFolderPath, "ListEnsemble_All__" + std::to_string(i) + "__", ""};
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
				m_savePngElement.launch(image3, m_pngNameSource);
				i++;
			}




			auto image3 = visualization3.createEnsembleVisualization(*ensemble, 500, 500, m_loadingWorkflow.getTraningSet(), m_joined_T_V, m_loadingWorkflow.getTestSet());
			SvmWokrflowConfiguration config_copy3{ "", "", "", m_config.outputFolderPath, "VotingEnsemble_All__", "" };
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, config_copy3);
			m_savePngElement.launch(image3, m_pngNameSource);
		}

		BaseSvmChromosome to_test;
		to_test.updateClassifier(ensemble);
		std::vector<BaseSvmChromosome> vec;
		vec.emplace_back(to_test);
		Population<BaseSvmChromosome> pop{ std::move(vec) };
		//TODO 10.02.2024 Remove AUC calculation at all as it is wrong (never think of how to properly join multiple SVMs in cascade for AUC calculation)
		//Calculate AUC for end report - TODO fix parrarell computation
		m_validation.launchSingleThread(pop, m_loadingWorkflow.getValidationSet()); //pop has 1 classifier so inside AUC calculation there is parallelism
		auto copy = pop;
		m_validationTest.launchSingleThread(copy, m_loadingWorkflow.getTestSet()); //pop has 1 classifier so inside AUC calculation there is parallelism

		auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
		auto bestOneTestMatrix = copy[pop.getBestIndividualIndex()].getConfusionMatrix().value();
		auto featureNumber = m_loadingWorkflow.getValidationSet().getSamples()[0].size();

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

		if(m_config.verbosity != platform::Verbosity::None)
		{
			m_resultLogger.logToFile(m_resultFilePath);
		}

		return ensemble;
	}
	catch (const std::runtime_error& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
		throw;
	}
}

void BigSetsEnsemble::logResults(std::shared_ptr<phd::svm::EnsembleListSvm> tree)
{
	BaseSvmChromosome to_test;
	to_test.updateClassifier(tree);
	std::vector<BaseSvmChromosome> vec;
	vec.emplace_back(to_test);
	Population<BaseSvmChromosome> pop{ std::move(vec) };
	//Calculate AUC for end report
	m_validation.launchSingleThread(pop, m_loadingWorkflow.getValidationSet()); //pop has 1 classifier so inside AUC calculation there is parallelism
	auto copy = pop;
	m_validationTest.launchSingleThread(copy, m_loadingWorkflow.getTestSet()); //pop has 1 classifier so inside AUC calculation there is parallelism

	auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
	auto bestOneTestMatrix = copy[pop.getBestIndividualIndex()].getConfusionMatrix().value();
	auto featureNumber = m_loadingWorkflow.getValidationSet().getSamples()[0].size();

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


	auto resultFilePath = m_resultFilePath;
	resultFilePath += "_all_trees.txt";
	if(m_config.verbosity != platform::Verbosity::None)
	{
		m_resultLogger.logToFile(resultFilePath);
	}
}

std::pair<double, int> BigSetsEnsemble::evaluateEnsemble(const dataset::Dataset<std::vector<float>, float>& dataset, bool testSet,
                                                         bool levelWiseScheme, std::shared_ptr<phd::svm::VotingEnsemble> ensemble)
{
	auto samples = dataset.getSamples();
	auto labels = dataset.getLabels();
	
	std::array<std::array<uint32_t, 2>, 2> matrix_ensemble = { 0 };
	auto number_of_uncertain = 0;


#pragma omp parallel for
	for (long long i = 0; i < static_cast<long long>(dataset.size()); i++)
	{
		float class_score = -1000;
		if (levelWiseScheme)
		{
			class_score = ensemble->classifyNodeWeights(samples[i]);
		}
		else
		{
			class_score = ensemble->classifyWithCertainty(samples[i]);
		}
		
		if (class_score != -100)
		{
			#pragma omp critical
			++matrix_ensemble[static_cast<int>(class_score)][static_cast<int>(labels[i])];
		}
		if (class_score == -100)
		{
			#pragma omp critical
			number_of_uncertain++;
		}
	}

	ConfusionMatrix metric_ensemble(matrix_ensemble[0][0], matrix_ensemble[1][1], matrix_ensemble[1][0], matrix_ensemble[0][1]);
	auto mcc_ensemble_score = metric_ensemble.MCC();


	if(testSet)
	{
		LOG_F(ERROR, "TEST Ensemble mcc score: %f  uncertain: %d   Test size: %d", mcc_ensemble_score, number_of_uncertain, dataset.size());
		LOG_F(ERROR, "TEST Ensemble confusion matrix (TP, FP, TN, FN): %s", metric_ensemble.to_string().c_str());
	}
	else
	{
		LOG_F(ERROR, "T+V Ensemble mcc score: %f  uncertain: %d   Tr+V size: %d", mcc_ensemble_score, number_of_uncertain, dataset.size());
		LOG_F(ERROR, "T+V Ensemble confusion matrix (TP, FP, TN, FN): %s", metric_ensemble.to_string().c_str());
	}

	
	return { mcc_ensemble_score, number_of_uncertain };
}

void BigSetsEnsemble::train(const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/)
{
	LOG_F(ERROR, "Start training dataset: %ls", m_config.trainingDataPath.c_str());
	
	// sample random validation 1000 vector or 20% of Tr + V
	std::call_once(m_initValidationSize, [&]()
		{
			auto classCount = svmUtils::countLabels(2, m_joined_T_V);

			if (std::all_of(classCount.begin(), classCount.end(), [](auto value)
				{
					return value > 0;
				}))
			{
				constexpr auto max_val = 1000.0;

				auto percent = max_val / static_cast<double>(classCount[0] + classCount[1]);
				if (percent > 0.20)
				{
					percent = 0.20;
				}

				auto IR_ratio = static_cast<double>(*std::max_element(classCount.begin(), classCount.end())) / static_cast<double>(*std::min_element(classCount.begin(), classCount.end()));

				if (IR_ratio > 5)
				{
					auto minorityClass = std::distance(classCount.begin(), std::min_element(classCount.begin(), classCount.end()));
					auto mcCount = static_cast<unsigned>(std::round(classCount[minorityClass] * 0.5));

					if(minorityClass == 0)
					{
						m_validationNegative = mcCount;
						//m_validationPositive = mcCount;
						m_validationPositive = static_cast<unsigned>(std::round(classCount[1] * percent));
					}
					else if (minorityClass == 1)
					{
						m_validationNegative = static_cast<unsigned>(std::round(classCount[0] * percent));
						//m_validationNegative = mcCount;
						m_validationPositive = mcCount;
					}
				}
				else
				{
					m_validationNegative = static_cast<unsigned>(std::round(classCount[0] * percent));
					m_validationPositive = static_cast<unsigned>(std::round(classCount[1] * percent));
				}
			}
			else
			{
				throw std::runtime_error("Only single class provided in dataset");
			}
		});


	std::vector<uint64_t> set;
	for (auto i = 0u; i < m_joined_T_V.size(); ++i)
	{
		set.emplace_back(i);
	}
	std::vector<uint64_t> vset;

	dataset::Dataset<std::vector<float>, float> empty;
	

	auto number_of_ensembles = 0;
	//auto max_ensembles = 1;
	auto seed = 1;
	auto [tr_l, trIds, val_l, validationIDS] = resample(m_joined_T_V, set, seed);

	auto mcc_compare = 0.0;
	auto uncertain_percent = 100.0;
	bool mcc_stop = false;
	bool uncertain_stop = false;
	
	//while(number_of_ensembles < max_ensembles /*|| tr_l.size() < 100 || val_l.size() < 500*/)
	while (true /*|| tr_l.size() < 100 || val_l.size() < 500*/)
	{
		LOG_F(INFO, "Size of training set: %d  for cascade:  %d", tr_l.size(), number_of_ensembles);
		LOG_F(INFO, "Size of validation set: %d",  val_l.size());
		
		auto temp = std::make_shared<phd::svm::ListNodeSvm>(nullptr);
		//build ensemble in here
		root = trainHelperNewDatasetFlow(temp, tr_l, trIds, val_l, validationIDS);

		//TODO check classification scheme
		std::shared_ptr<phd::svm::EnsembleListSvm> tree = std::make_shared<phd::svm::EnsembleListSvm>(root, m_listLength, m_newClassificationScheme);
		m_allEnsembles.emplace_back(tree);
		m_listLength = 0;

		//evaluate on Tr + V
		//logResults(tree);

		//ConfusionMatrix metric(*tree, m_joined_T_V);
		//auto mcc_score = metric.MCC();
		//
		//
		std::array<std::array<uint32_t, 2>, 2> matrix = { 0 };



		auto& evaluationSet = m_loadingWorkflow.getValidationSet();
		std::vector<std::int64_t> wrongOnesIds(evaluationSet.size(), -1);
		
		auto samples = evaluationSet.getSamples();
		auto labels = evaluationSet.getLabels();

#pragma omp parallel for
		for(long long i = 0; i < static_cast<long long>(evaluationSet.size()); i++)
		{
			auto class_score = tree->classifyWithCertainty(samples[i]);
			if(class_score != -100)
			{
				#pragma omp critical
				++matrix[static_cast<int>(class_score)][static_cast<int>(labels[i])];
			}
			if (class_score == -100)
			//if (class_score != labels[i])
			{
				wrongOnesIds[i] = i;
			}
		}

		wrongOnesIds.erase(std::remove_if(wrongOnesIds.begin(), wrongOnesIds.end(),
			[](const int64_t o)
			{
				return o == -1;
			}), wrongOnesIds.end());

		std::vector<std::uint64_t> wrongOnesIds2;
		wrongOnesIds2.reserve(wrongOnesIds.size());
		for(auto val : wrongOnesIds)
		{
			wrongOnesIds2.emplace_back(static_cast<std::uint64_t>(val));
		}
		

		ConfusionMatrix metric(matrix[0][0], matrix[1][1], matrix[1][0], matrix[0][1]);
		auto mcc_score = metric.MCC();
		m_weights.emplace_back(mcc_score);
		m_scores.emplace_back(mcc_score);
		//m_weights.emplace_back(1);


		LOG_F(ERROR, "Single list mcc score: %f  wrong one size: %d   Tr+V size: %d", mcc_score, wrongOnesIds.size(), m_joined_T_V.size());


		
		
		//create dataset from ids
		auto wrongOnesSet = createDatasetFromIds(m_joined_T_V, wrongOnesIds2);


		

		//build compensation
		//
		//dataset::Dataset<std::vector<float>, float> tr_l2;
		//std::vector<unsigned long long> trIds2;
		//dataset::Dataset<std::vector<float>, float> val_l2;
		//std::vector<unsigned long long> validationIDS2;
		//
		//if (number_of_ensembles % 2 == 1 && wrongOnesIds.size() > 32) //
		//{
		//	auto resampled = resample(wrongOnesSet, wrongOnesIds2, seed);
		//	tr_l2= std::get<0>(resampled);
		//	trIds2 = std::get<1>(resampled);
		//	val_l2 = std::get<2>(resampled);
		//	validationIDS2 = std::get<3>(resampled);
		//	
		//}
		//else
		//{
		//	auto resampled = resample(m_joined_T_V, set, (number_of_ensembles * 100));
		//	tr_l2 = std::get<0>(resampled);
		//	trIds2 = std::get<1>(resampled);
		//	val_l2 = std::get<2>(resampled);
		//	validationIDS2 = std::get<3>(resampled);
		//}
		
		//select new dataset for validation, and training
		//auto [tr_l2, trIds2, val_l2, validationIDS2] = resample(wrongOnesSet, wrongOnesIds2, seed);
		auto [tr_l2, trIds2, val_l2, validationIDS2] = resample(m_joined_T_V, set, (number_of_ensembles * 100));

		tr_l = tr_l2;
		trIds = trIds2;
		val_l = val_l2;
		validationIDS = validationIDS2;

		number_of_ensembles++;





		std::shared_ptr<phd::svm::VotingEnsemble> ensemble = std::make_shared<phd::svm::VotingEnsemble>(m_allEnsembles, m_weights);
		bool levelWiseClassificationScheme = true;

		if(levelWiseClassificationScheme)
		{
			ensemble->scoreLevelWise(m_joined_T_V);
		}
		
		auto [mcc_ensemble, uncertain_number] = evaluateEnsemble(m_joined_T_V, false, levelWiseClassificationScheme, ensemble); //ORIGINAL
		//auto [mcc_ensemble, uncertain_number] = evaluateEnsemble(m_loadingWorkflow.getValidationSet(), false, levelWiseClassificationScheme, ensemble); //NEW TEST FOR STOP CONDITION
		evaluateEnsemble(m_loadingWorkflow.getTestSet(), true, levelWiseClassificationScheme, ensemble);
		auto current_uncertain = static_cast<double>(uncertain_number) / static_cast<double>(m_joined_T_V.size());


		//-------------------------------------------------NO NEED FOR ODD NUMBER OF CLASSIFIERS IF WE USE MCC SCORE FOR WEIGHTS
		if (number_of_ensembles % 2 == 0) //only odd number should be evaluated
		{
			continue;
		}
		
		if (mcc_ensemble - mcc_compare > 0.001 || uncertain_percent - current_uncertain > 1.0 || number_of_ensembles < 4)
		{
			
			mcc_compare = mcc_ensemble;
			uncertain_percent = current_uncertain;
		}
		else
		{
			mcc_stop = ! (mcc_ensemble - mcc_compare > 0.001);
			uncertain_stop = ! (uncertain_percent - current_uncertain > 1.0);
			// delete the last element in the list of cascades as it didn't improve overall performance
			m_allEnsembles.pop_back();
			m_weights.pop_back();
			
			break;
		}
	}
	

	auto btos = [&](bool cond)
	{
		return cond ? "true" : "false";
	};
	LOG_F(INFO, "Stop conditions: mcc_stop=%s,   uncertain_stop=%s", btos(mcc_stop), btos(uncertain_stop));
	LOG_F(INFO, "Number of ensembles: %d", number_of_ensembles);
	LOG_F(INFO, "Number of linears %d,  number of RBF %d", m_numberOfLinearSVM, m_numberOfRbfSVM);
	
	LOG_F(ERROR, "End training dataset: %ls", m_config.trainingDataPath.c_str());
}


std::shared_ptr<phd::svm::ListNodeSvm> BigSetsEnsemble::trainHelperNewDatasetFlow(std::shared_ptr<phd::svm::ListNodeSvm>& root_,
	const dataset::Dataset<std::vector<float>, float>& trainingSet,
	const std::vector<uint64_t>& ids,
	const dataset::Dataset<std::vector<float>, float>& validationSet,
	const std::vector<uint64_t>& valIds)
{
	try
	{
		m_id = 0;
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
		dataset::Dataset<std::vector<float>, float> tr_l = trainingSet;
		dataset::Dataset<std::vector<float>, float> val_l = validationSet;
		dataset::Dataset<std::vector<float>, float> test_l = m_loadingWorkflow.getTestSet();
		std::vector<uint64_t> trIds = ids;
		std::vector<uint64_t> validationIDS = valIds;

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
		for (auto i = 0u; i < m_loadingWorkflow.getTestSet().size(); ++i)
		{
			testSetIds.emplace_back(i);
		}

		std::vector<uint64_t> countTR = countClasses(trainingSet);

		auto temp = root_;
		auto stopCondition = [&]()
		{
			return (nothingImprovedInFullSet || m_id >= max_length || datasetTooSmall);
		};

		bool cascadeWideFeatureSelection = m_full_config.getValue<bool>("Svm.EnsembleTree.UseFeatureSelctionCascadeWise");
		
		while (!stopCondition()) //limit size of ensemble for visualization speed problems
		{
			auto numberOfFeatures = m_loadingWorkflow.getValidationSet().getSample(0).size();
			if (cascadeWideFeatureSelection && numberOfFeatures > 8)
			{
				std::vector<Feature> featuresSet;
				
				auto randomNumberOfFeatures = std::uniform_int_distribution<int>(2, static_cast<int>(numberOfFeatures));
				auto features = std::uniform_int_distribution<int>(0, static_cast<int>(numberOfFeatures));
				auto rngEngine = std::make_unique<my_random::MersenneTwister64Rng>(0); //TODO change for regular algorithm seed after tests

				auto selectedFeatures = rngEngine->getRandom(randomNumberOfFeatures);
				for(auto i = 0; i < selectedFeatures; ++i)
				{
					featuresSet.emplace_back(rngEngine->getRandom(features));
				}

				SvmFeatureSetMemeticChromosome individual(std::move(featuresSet));
				tr_l = individual.convertChromosome(tr_l);
				val_l = individual.convertChromosome(val_l);
				test_l = individual.convertChromosome(m_loadingWorkflow.getTestSet());
			}
			bool linearIsBetter = false;
			DatasetLoaderHelper datasets(tr_l, val_l, test_l);
			
			{
				try
				{
					//std::cout << "ID:" << m_listLength << "\n";
					auto configurationForNode = EnsembleTreeWorkflowConfig(m_full_config, datasets);

					bool newDatasetFlow = true;
					if (m_full_config.getValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet"))
					{
						newDatasetFlow = false; //see SvmHelper Evaluate function to see wh it is false
					}


					BigSetsSvmHelper helper(m_config, configurationForNode, datasets, m_algorithmConfig.m_addSvToTraining,
						svVectorsToPassDown, m_loadingWorkflow, svVectorsWithGammas,
						m_useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config, cascadeWideFeatureSelection, newDatasetFlow);
					svm = helper.run();


					/*BigSetsSvmHelper helper4(m_config, configurationForNode, datasets, false,
						svVectorsToPassDown, m_loadingWorkflow, svVectorsWithGammas,
						false, m_debugLog, m_useFeatureSelection, m_full_config, newDatasetFlow);
					auto svm_no_inheritance = helper4.run();*/


					auto config_copy = m_full_config;
					config_copy.putValue("Svm.KernelType", "LINEAR");
					config_copy.putValue("Svm.EnsembleTree.DasvmKernel", false);
					auto useDasvmKernel = false;
					auto configurationForNodeLinear = EnsembleTreeWorkflowConfig(config_copy, datasets);
					BigSetsSvmHelper helper2(m_config, configurationForNodeLinear, datasets, false,
						svVectorsToPassDown, m_loadingWorkflow, svVectorsWithGammas,
						useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config, cascadeWideFeatureSelection, newDatasetFlow);
					auto linearsvm = helper2.run();



				/*	config_copy.putValue("Svm.KernelType", "POLY");
					config_copy.putValue("Svm.EnsembleTree.DasvmKernel", false);
					auto configurationForNodePoly = EnsembleTreeWorkflowConfig(config_copy, datasets);
					BigSetsSvmHelper helper3(m_config, configurationForNodePoly, datasets, false,
						svVectorsToPassDown, m_loadingWorkflow, svVectorsWithGammas,
						useDasvmKernel, m_debugLog, m_useFeatureSelection, m_full_config, newDatasetFlow);
					auto polysvm = helper3.run();*/

					/*if (helper4.getBestOne().getFitness() > helper.getBestOne().getFitness())
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

					}*/
					if (helper2.getBestOne().getFitness() > helper.getBestOne().getFitness())
					{
						LOG_F(INFO, "Linear is better %.5f  RBF: %.5f", helper2.getBestOne().getFitness(), helper.getBestOne().getFitness());
						svm = linearsvm;
						linearIsBetter = true;
						m_numberOfLinearSVM++;
					}
					else
					{
						m_numberOfRbfSVM++;
					}

					//if (helper3.getBestOne().getFitness() > helper.getBestOne().getFitness() && helper3.getBestOne().getFitness() > helper2.getBestOne().getFitness())
					//{
					//	LOG_F(INFO, "Poly is better %.5f  RBF: %.5f", helper3.getBestOne().getFitness(), helper.getBestOne().getFitness());
					//	svm = polysvm;
					//	linearIsBetter = true;
					//}

					//auto tempNone = std::shared_ptr<phd::svm::ISvm>();
					//svm = runGridSearch(datasets, tempNone);
					//
					temp->m_svm = svm;
				}
				catch (const std::runtime_error& e)
				{
					LOG_F(ERROR, "Error: %s", e.what());
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

					//createVisualizationNewFlow(tempTR2, m_joined_T_V, m_id, svm, tempTR_IDS, validationIDS);

				}
				else
				{
					//createVisualization(tr_l, val_l, m_id, svm, trIds, validationIDS);

					if (m_full_config.getValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet"))
					{
						//USE FULL VALIDATION IN HERE
						//createVisualizationNewFlow(fullSet, joinSets(fullSet, m_loadingWorkflow.getValidationSet()), m_id, svm, fullSetIds, fullSetIds);
					}
					else
					{
						//createVisualizationNewFlow(fullSet, fullSet, m_id, svm, fullSetIds, fullSetIds);
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


			//pararelize this code with uncertainDataset
			auto [uncertainTrainingSet, uncertainIds] = getUncertainDataset(tr_l, trIds, svm);
			auto [uncertainValidationSet, uncertainIdsValidation] = getUncertainDataset(val_l, validationIDS, svm);
			previousUncertain = uncertainTrainingSet;

			auto [uncertainFullSet, uncertainIdsFullSet] = getUncertainDataset(fullSet, fullSetIds, svm);
			auto countFullSet = countClasses(uncertainFullSet);
			nothingImprovedInFullSet = fullSet.size() == uncertainFullSet.size();
			svmComponents::SvmTrainingSetChromosome uncertainFullSetCh{ std::move(uncertainFullSet) };
			fullSet = uncertainFullSetCh.convertChromosome(m_joined_T_V);
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
			tr_l = uncertainTraining.convertChromosome(m_joined_T_V);
			svmComponents::SvmTrainingSetChromosome uncertainValidation{ std::move(uncertainValidationSet) };
			val_l = uncertainValidation.convertChromosome(m_joined_T_V);

			trIds = uncertainIds;
			validationIDS = uncertainIdsValidation;


			if (m_debugLog)
			{
				auto [uncertainTestSet, uncertainIdsTest] = getUncertainDataset(test_l, testSetIds, svm);
				svmComponents::SvmTrainingSetChromosome uncertainTest{ std::move(uncertainTestSet) };
				test_l = uncertainTest.convertChromosome(m_loadingWorkflow.getTestSet());
				testSetIds = uncertainIdsTest;

				//saveRegions(fullSetIds, fullSetIds, testSetIds, true);
			}

			m_listLength++;
			m_id++;
			if (stopCondition())
			{
				break;
			}

			try
			{
				//if (!tr_l.empty() || !val_l.empty())
				{
					//RESAMPLE AGAIN FROM UNCERTAIN SETS, remember about ids
					//auto result = resample(tr_l, trIds, val_l, validationIDS);

					//tr_l = std::get<0>(result);
					//trIds = std::get<1>(result);
					//val_l = std::get<2>(result);
					//validationIDS = std::get<3>(result);
				}
			}
			catch (const std::runtime_error& e)
			{
				LOG_F(ERROR, "Error: %s", e.what());
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
		LOG_F(INFO, "Size of uncertain validation %d,  original size %d", val_l.size(), m_loadingWorkflow.getValidationSet().size());
		LOG_F(INFO, "Size of uncertain training %d,  original size %d", tr_l.size(), m_loadingWorkflow.getTraningSet().size());


		return root_;
	}
	catch (const std::runtime_error& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
		throw;
	}

}



std::tuple<dataset::Dataset<std::vector<float>, float>, std::vector<unsigned long long>,
	dataset::Dataset<std::vector<float>, float>, std::vector<unsigned long long>>
	BigSetsEnsemble::resample(
		const dataset::Dataset<std::vector<float>, float>& dataset,
		const std::vector<unsigned long long>& ids, int seed)
{
	auto joined_tr_val = dataset;
	auto joined_ids = ids;
	
	std::unordered_set<uint64_t> indexsSet;
	dataset::Dataset<std::vector<float>, float> newValidation;
	std::vector<unsigned long long>  newValidationIds;

	auto rngEngine = std::make_unique<my_random::MersenneTwister64Rng>(seed);
	

	auto classCount = svmUtils::countLabels(2, joined_tr_val);

	int minTrainingSizeForClass = m_algorithmConfig.m_trainingSetOptimization->getInitialTrainingSetSize();
	int increasedTrainingSize = 8;


	//if (!m_full_config.getValue<bool>("Svm.EnsembleTree.ResamplingWithNoAddition"))
	{
		long class0 = classCount[0];
		long class1 = classCount[1];

	
		//make sure that there will be enough vectors to run genetic algorithm
		if(class0 - increasedTrainingSize * minTrainingSizeForClass > static_cast<long>(m_validationNegative))
		{
			class0 -= increasedTrainingSize * minTrainingSizeForClass;
		}
		if (class1 - increasedTrainingSize * minTrainingSizeForClass > static_cast<long>(m_validationPositive))
		{
			class1 -= increasedTrainingSize * minTrainingSizeForClass;
		}
	
		//add samples from full sets if there not enough examples
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
	//else
	//{
	//	//make sure that there will be enough vectors to run genetic algorithm
	//	if (static_cast<int>(classCount[0] - m_validationNegative) < minTrainingSizeForClass)
	//	{
	//		m_validationNegative -= increasedTrainingSize * minTrainingSizeForClass - (classCount[0] - m_validationNegative);
	//	}
	//	if (static_cast<int>(classCount[1] - m_validationPositive) < minTrainingSizeForClass)
	//	{
	//		m_validationPositive -= increasedTrainingSize * minTrainingSizeForClass - (classCount[1] - m_validationPositive);
	//	}
	//}


	auto randomID = std::uniform_int_distribution<int>(0, static_cast<int>(joined_tr_val.size() - 1));
	
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

	auto size = newTraining.size();
	LOG_F(INFO, "Resampling size of training %d, size of validation %d", size, newValidation.size());

	return std::make_tuple(newTraining, newTrainingIds, newValidation, newValidationIds);
}

} // namespace genetic
