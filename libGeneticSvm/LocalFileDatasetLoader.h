#pragma once

#include <filesystem>
//#include "libFileSystem/FileSystemDefinitions.h"
#include "libDataset/Dataset.h"
#include "libGeneticSvm/IDatasetLoader.h"
#include "libSvmStrategies/NormalizeStrategy.h"
//#include "libStrategies/TabularDataProviderStrategy.h"

namespace genetic
{
class LocalFileDatasetLoader final : public IDatasetLoader
{
public:
	LocalFileDatasetLoader(const std::filesystem::path& trainingSetPath,
	                       const std::filesystem::path& validationSetPath,
	                       const std::filesystem::path& testSetPath,
	                       bool normalize = true,
						   bool resampleTrainingValidation = false);

    explicit LocalFileDatasetLoader(const std::filesystem::path& datasetPath);

    LocalFileDatasetLoader(const std::filesystem::path& trainingSetPath,
                           const std::filesystem::path& validationSetPath,
                           const std::filesystem::path& testSetPath,
                           std::vector<bool> featureMask);

    const dataset::Dataset<std::vector<float>, float>& getTraningSet() override;
    const dataset::Dataset<std::vector<float>, float>& getValidationSet() override;
    const dataset::Dataset<std::vector<float>, float>& getTestSet() override;
    bool isDataLoaded() const override;
    const std::vector<float>& scalingVectorMin() override;
    const std::vector<float>& scalingVectorMax() override;

private:
    void loadDataAndNormalize();

    svmStrategies::NormalizeStrategy m_normalizeElement;

    std::once_flag m_shouldLoadData;
    bool m_isDataLoaded;
    bool m_normalize;
    dataset::Dataset<std::vector<float>, float> m_traningSet;
    dataset::Dataset<std::vector<float>, float> m_validationSet;
    dataset::Dataset<std::vector<float>, float> m_testSet;

    std::vector<float> m_scallingMin;
    std::vector<float> m_scallingMax;

    const std::filesystem::path m_trainingSetPath;
    const std::filesystem::path m_validationSetPath;
    const std::filesystem::path m_testSetPath;

    std::vector<bool> m_featureMask;
	bool m_resampleTrainingValidation;
};
} // namespace genetic
