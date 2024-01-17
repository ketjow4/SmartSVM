#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IDatasetLoader.h"

#include "libStrategies/FileSinkStrategy.h"
//#include "AllModelsLogger.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "SvmLib/EnsembleListSvm.h"

//#include "libSvmComponents/SvmMetricFactory.h"

namespace genetic
{
class EnsembleTreeWorkflow : public ISvmAlgorithm
{
public:
	explicit EnsembleTreeWorkflow(const SvmWokrflowConfiguration& config,
	                              EnsembleTreeWorkflowConfig algorithmConfig,
	                              IDatasetLoader& workflow,
								  platform::Subtree full_config);

	std::shared_ptr<phd::svm::ISvm> run() override;

	void update();

private:

	void EnsembleTreeWorkflow::createVisualization(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                               const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                               int& id_,
	                                               std::shared_ptr<phd::svm::ISvm> svm,
	                                               const std::vector<uint64_t>& trIds,
	                                               const std::vector<uint64_t>& valIds);

	void createVisualizationNewFlow(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                int& id_,
	                                std::shared_ptr<phd::svm::ISvm> svm,
	                                const std::vector<uint64_t>& trIds,
	                                const std::vector<uint64_t>& valIds);
	
		
	void addLastNodeWithFullSvm(std::shared_ptr<phd::svm::ListNodeSvm>& temp);

	void train(const dataset::Dataset<std::vector<float>, float>& trainingSet);

	void saveRegions(const std::vector<unsigned long long>& trainUncertainIds, 
		const std::vector<unsigned long long>& validationIds,
		const std::vector<unsigned long long>& testSetIds,
		bool useTrainedWithValidation = false) const;
	std::shared_ptr<phd::svm::ListNodeSvm> trainHelper(std::shared_ptr<phd::svm::ListNodeSvm>& root_,
	                                                   const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                   const std::vector<uint64_t>& ids,
	                                                   const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                                   const std::vector<uint64_t>& valIds);

	std::shared_ptr<phd::svm::ListNodeSvm> trainHelperNewDatasetFlow(std::shared_ptr<phd::svm::ListNodeSvm>& root_,
	                                                                 const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                                 const std::vector<uint64_t>& ids,
	                                                                 const dataset::Dataset<std::vector<float>, float>& validationSet,
	                                                                 const std::vector<uint64_t>& valIds);

	std::tuple<dataset::Dataset<std::vector<float>, float>, std::vector<unsigned long long>,
	           dataset::Dataset<std::vector<float>, float>, std::vector<unsigned long long>>
	resample(const dataset::Dataset<std::vector<float>, float>& dataset,
	         const std::vector<unsigned long long>& ids,
	         const dataset::Dataset<std::vector<float>, float>& validationSet,
	         const std::vector<unsigned long long>& valIds);
	
	dataset::Dataset<std::vector<float>, float> EnsembleTreeWorkflow::getCertain(std::shared_ptr<phd::svm::ISvm> svm,
		const dataset::Dataset<std::vector<float>, float>& set, bool certain);
	
	std::pair<dataset::Dataset<std::vector<float>, float>,  dataset::Dataset<std::vector<float>, float>>
	scoreEnsemble(std::shared_ptr<phd::svm::EnsembleListSvm> ensemble);

	void chartDataSave(std::shared_ptr<phd::svm::libSvmImplementation> svm, int list_length,   
		const dataset::Dataset<std::vector<float>, float>& val,
		const dataset::Dataset<std::vector<float>, float>& test);

	void perNodeVisualization(std::shared_ptr<phd::svm::EnsembleListSvm> ensemble, int length);
	
	std::shared_ptr<phd::svm::ISvm> runGridSearch(IDatasetLoader& workflow, std::shared_ptr<phd::svm::ISvm>& previousNode);

	std::filesystem::path m_resultFilePath;
	EnsembleTreeWorkflowConfig m_algorithmConfig;
    SvmWokrflowConfiguration m_config;
	IDatasetLoader* m_loadingWorkflow;
    platform::Subtree m_full_config;

    std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_metric;
    svmStrategies::SvmValidationStrategy<svmComponents::BaseSvmChromosome> m_validation;
    svmStrategies::SvmValidationStrategy<svmComponents::BaseSvmChromosome> m_validationTest;
	
	strategies::FileSinkStrategy m_savePngElement;
	std::filesystem::path m_outputPath;

    GeneticWorkflowResultLogger m_resultLogger;

	Timer m_timer;

	std::once_flag m_initValidationSize;
	unsigned int m_validationPositive;
	unsigned int m_validationNegative;


	std::shared_ptr<phd::svm::ListNodeSvm> root;

	//const SvmWokrflowConfiguration& m_config;
	//IDatasetLoader& m_loadingWorkflow;
	//EnsembleTreeWorkflowConfig m_algorithmConfig;
	//platform::Subtree m_full_config;
	int m_listLength;
	int m_id;

	bool m_newClassificationScheme;
	bool m_useDasvmKernel;
	bool m_debugLog;
	bool m_useFeatureSelection;
	dataset::Dataset<std::vector<float>, float> m_joined_T_V;

	dataset::Dataset<std::vector<float>, float> m_updateTr;
	dataset::Dataset<std::vector<float>, float> m_updateVal;
	dataset::Dataset<std::vector<float>, float> m_updateTest;

	static constexpr const char* m_algorithmName = "EnsembleTree";
	//logger::LogFrontend m_logger;
};
} // namespace genetic
