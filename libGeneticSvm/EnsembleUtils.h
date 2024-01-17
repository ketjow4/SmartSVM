#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IDatasetLoader.h"

#include "libStrategies/FileSinkStrategy.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "SvmLib/EnsembleListSvm.h"



namespace genetic
{
	std::pair<std::vector<DatasetVector>, std::vector<uint64_t>> getSVsIds(const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const std::vector<uint64_t>& ids,
		std::shared_ptr<phd::svm::ISvm> svm);

	dataset::Dataset<std::vector<float>, float> createDatasetFromIds(const dataset::Dataset<std::vector<float>, float>& dataset,
	                                                                 std::vector<uint64_t>& manualDataset);

	std::pair<std::vector<svmComponents::DatasetVector>, std::vector<uint64_t>>
	getUncertainDataset(const dataset::Dataset<std::vector<float>, float>& trainingSet, const std::vector<uint64_t>& ids, std::shared_ptr<phd::svm::ISvm> svm);

	std::pair<std::vector<svmComponents::DatasetVector>, std::vector<uint64_t>>
	getCertainDataset(const dataset::Dataset<std::vector<float>, float>& trainingSet, const std::vector<uint64_t>& ids, std::shared_ptr<phd::svm::ISvm> svm);

	std::pair<std::vector<svmComponents::DatasetVector>, std::vector<uint64_t>> getSVsIds(const dataset::Dataset<std::vector<float>, float>& trainingDataset,
	                                                                                      const std::vector<uint64_t>& ids,
	                                                                                      std::shared_ptr<phd::svm::ISvm> svm);

	std::vector<uint64_t> countClasses(const dataset::Dataset<std::vector<float>, float>& trainingDataset, unsigned int numberOfClasses = 2);

	std::vector<uint64_t> countClasses(const std::vector<DatasetVector>& trainingDataset, unsigned int numberOfClasses = 2);

	bool is_empty(std::string& filename);
} // namespace genetic
