#include "NormalizeStrategy.h"
#include "libSvmComponents/DataNormalization.h"

namespace svmStrategies
{
std::string NormalizeStrategy::getDescription() const
{
    return "Normalization of datasets for use of SVM classifier";
}

std::tuple<dataset::Dataset<std::vector<float>, float>&,
           dataset::Dataset<std::vector<float>, float>&,
           dataset::Dataset<std::vector<float>, float>&,
           std::vector<float>,
           std::vector<float>> NormalizeStrategy::launch(dataset::Dataset<std::vector<float>, float>& trainingData,
                                                         dataset::Dataset<std::vector<float>, float>& validationData,
                                                         dataset::Dataset<std::vector<float>, float>& testData)
{
    const auto featuresCount = trainingData.getSample(0).size();

    if(m_normalize)
    {
        svmComponents::DataNormalization normalization(static_cast<unsigned int>(featuresCount));
        normalization.normalize(trainingData, validationData, testData);
        auto min = normalization.minValuesOfFeatures();
		auto max = normalization.maxValuesOfFeatures();

        return std::forward_as_tuple(trainingData, validationData, testData, min, max);
    }
    else
    {
    	//This is default option
        svmComponents::StandardScaler normalization(static_cast<unsigned int>(featuresCount));
        normalization.normalize(trainingData, validationData, testData);

        auto min = normalization.getMeanVector();
        auto max = normalization.getStdDevV();

        return std::forward_as_tuple(trainingData, validationData, testData, min, max);
    }
}
} // namespace svmStrategies
