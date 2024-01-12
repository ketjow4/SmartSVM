
#include <numeric>
#include "libDataset/Dataset.h"
#include "libSvmComponents/R2_regression.h"
#include "BaseSvmChromosome.h"
#include "SvmComponentsExceptions.h"

namespace svmComponents
{
Metric R2_regression::calculateMetric(const BaseSvmChromosome& individual, 
                                      const dataset::Dataset<std::vector<float>, float>& testSamples) const
{
    if (testSamples.empty())
    {
        throw EmptyDatasetException(DatasetType::ValidationOrTest);
    }

    auto svmModel = individual.getClassifier();
    if (!svmModel->isTrained())
    {
        throw UntrainedSvmClassifierException();
    }

    auto samples = testSamples.getSamples();
    auto targets = testSamples.getLabels();
    //int sampleNumber = 0;

    std::vector<float> scores;
    for(auto& sample : samples)
    {
        scores.emplace_back(svmModel->classify(sample));
    }

    //r2 = list(map(lambda p: R2(p, valid['winRate']), pred_list))

    //def R2(x, y) :
    //    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y)))
    std::vector<float> a, b;
    float targetMean = static_cast<float>(std::accumulate(targets.begin(), targets.end(), 0.0) / targets.size());
    for(auto i = 0u; i < scores.size(); i++)
    {
        a.emplace_back(std::pow(targets[i] - scores[i], 2));
        b.emplace_back(std::pow(targets[i] - targetMean, 2));
    }
    float temp = static_cast<float>(std::accumulate(a.begin(), a.end(), 0.0));
    float temp2 = static_cast<float>(std::accumulate(b.begin(), b.end(), 0.0));
    return Metric((1 - temp / temp2));
}
} // namespace svmComponents
