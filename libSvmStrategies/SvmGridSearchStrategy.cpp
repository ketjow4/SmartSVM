//
//#include "libSvmComponents/SvmComponentsExceptions.h"
//#include "libFilesystem/FileSystemDefinitions.h"
//#include "SvmGridSearchStrategy.h"
//
//namespace svmStrategies
//{
//SvmGridSearchStrategy::SvmGridSearchStrategy() 
//{
//}
//
//std::string SvmGridSearchStrategy::getDescription() const
//{
//    return "This element do grid search for svm classifier based on the inputs. Internally uses Opencv svm implementation";
//}
//
//bool SvmGridSearchStrategy::isInitialized(const framework::InputReader inputReader) const
//{
//    return inputReader.getInput(0).empty() == true && inputReader.getInput(1).empty() == true && m_gridSearchAlgorithm != nullptr;
//}
//
//void SvmGridSearchStrategy::launch(framework::InputReader&& inputReader, framework::OutputWriter&& outputWriter)
//{
//    try
//    {
//        if(isInitialized(inputReader))
//        {
//            m_gridSearchAlgorithm->calculateGrids();
//            outputWriter.setOutput(0, m_gridSearchAlgorithm->calculate());
//            outputWriter.setOutput(1, m_gridSearchAlgorithm->getImage());
//            m_status = framework::ElementStrategyStatus::Success;
//        }
//        else
//        {
//            const auto validationData = boost::any_cast<dataset::Dataset<std::vector<float>, float>>(inputReader.getInput(0));
//            const auto testData = boost::any_cast<dataset::Dataset<std::vector<float>, float>>(inputReader.getInput(1));
//            auto config = boost::any_cast<svmComponents::GridSearchConfiguration*>(inputReader.getInput(2));
//            const auto outputFile = boost::any_cast<filesystem::Path>(inputReader.getInput(3));
//
//            m_gridSearchAlgorithm = std::make_unique<svmComponents::GridSearchImplementation>(*config, std::move(validationData), std::move(testData), outputFile);
//            outputWriter.setOutput(0, m_gridSearchAlgorithm->calculate());
//            outputWriter.setOutput(1, m_gridSearchAlgorithm->getImage());
//
//}
//} // namespace svmStrategies