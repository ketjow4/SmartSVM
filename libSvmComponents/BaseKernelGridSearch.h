
#pragma once

#include <opencv2/ml.hpp>
#include "libGeneticComponents/Population.h"
#include "SvmKernelChromosome.h"

namespace svmComponents
{
class BaseSvmChromosome;

class BaseKernelGridSearch
{
public:
    virtual ~BaseKernelGridSearch() = default;

    // @wdudzik for our grid search 
    virtual void calculateGrids(const BaseSvmChromosome& individual) = 0;
    virtual geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() = 0;

    // @wdudzik for OpenCV grid search 
    virtual void calculateGrids() = 0;
    virtual std::string logSvmParameters() = 0;
    virtual void performGridSearch(cv::Ptr<cv::ml::TrainData> trainingSet, unsigned int numberOfFolds) = 0;
    void setSvm(cv::Ptr<cv::ml::SVM> svm);

protected:
    cv::ml::ParamGrid calculateNewGrid(const cv::ml::ParamGrid& parametersGrid, double gridParemeter) const;
    static unsigned int calculateNumberOfSteps(const cv::ml::ParamGrid& parametersGrid);

    cv::Ptr<cv::ml::SVM> m_svm;
};

inline void BaseKernelGridSearch::setSvm(cv::Ptr<cv::ml::SVM> svm)
{
    m_svm = svm;
}
} // namespace svmComponents
