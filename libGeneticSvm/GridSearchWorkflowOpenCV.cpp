//
//#include <libStrategies/TabularDataProviderStrategy.h>
//#include <libSvmStrategies/SvmGridSearchStrategy.h>
//#include <libStrategies/FileSinkStrategy.h>
//#include "GridSearchWorkflowOpenCV.h"
//#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
//#include "WorkflowUtils.h"
//#include "IDatasetLoader.h"
//
//namespace genetic
//{
//void GridSearchWorkflowOpenCV::moveStrategiesToElements()
//{
//    using namespace framework;
//    m_valdiationProviderElement = createElement<strategies::TabularDataProviderStrategy>();
//    m_testProviderElement = createElement<strategies::TabularDataProviderStrategy>();
//    m_gridSearchElement = createElement<svmStrategies::SvmGridSearchStrategy>();
//    m_pngFileElement = createElement<strategies::FileSinkStrategy>();
//}
//
//GridSearchWorkflowOpenCV::GridSearchWorkflowOpenCV(const SvmWokrflowConfiguration& config,
//                                                   svmComponents::GridSearchConfiguration&& algorithmConfig,
//                                                   IDatasetLoader& workflow)
//    : m_config(config)
//    , m_workflow(workflow)
//    , m_algorithmConfig(std::move(algorithmConfig))
//{
//    moveStrategiesToElements();
//}
//
//void GridSearchWorkflowOpenCV::connectDataToSecondScheduler()
//{
//    try
//    {
//        auto validation = m_workflow.getValidationSet();
//        auto test = m_workflow.getTestSet();
//
//        m_validationSource.setValue(validation);
//        m_testSource.setValue(test);
//
//        m_gridSearchElement->setInputConnection(0, m_validationSource);
//        m_gridSearchElement->setInputConnection(1, m_testSource);
//    }
//    catch (const std::exception& exception)
//    {
//        m_logger.LOG(logger::LogLevel::Error, exception.what());
//    }
//}
//
//bool GridSearchWorkflowOpenCV::isNotLastIteration(unsigned int iteration) const
//{
//    return iteration < m_algorithmConfig.m_numberOfIterations;
//}
//
//void GridSearchWorkflowOpenCV::runGridSearch()
//{
//    try
//    {
//        std::vector<std::shared_ptr<framework::Element>> gridSerachWorkflow
//        {
//            m_gridSearchElement,
//            m_pngFileElement,
//        };
//
//        if (!m_algorithmConfig.m_svmConfig.m_doVisualization)
//        {
//            m_pngFileElement->enable(false);
//        }
//
//        framework::Scheduler gridScheduler(gridSerachWorkflow);
//
//        std::condition_variable gridFinished;
//        std::mutex gridFinishedMutex;
//
//        gridScheduler.registerObserver(framework::SchedulerEvent::WorkflowFinished,
//                                       [&](const auto bundle)
//                                       {
//                                           gridFinished.notify_one();
//                                       });
//
//        for (unsigned int i = 0; i < m_algorithmConfig.m_numberOfIterations; i++)
//        {
//            m_pngNameSource.reset();
//            m_pngNameSource.setValue(filesystem::Path(
//                m_config.outputFolderPath.string() +
//                m_config.visualizationFilename +
//                std::to_string(i) + ".png"));
//
//            gridScheduler.start();
//
//            std::unique_lock<std::mutex> gridLock(gridFinishedMutex);
//            gridFinished.wait(gridLock,
//                              [&gridScheduler]()
//                              {
//                                  return gridScheduler.isFinished();
//                              });
//
//            if (isNotLastIteration(i))
//            {
//                gridScheduler.reset();
//            }
//
//            m_validationSource.reset();
//            m_testSource.reset();
//        }
//    }
//    catch (const std::exception& exception)
//    {
//        m_logger.LOG(logger::LogLevel::Error, exception.what());
//    }
//}
//
//std::shared_ptr<phd::svm::ISvm> GridSearchWorkflowOpenCV::run()
//{
//    try
//    {
//        connectElements();
//        connectDataToSecondScheduler();
//        runGridSearch();
//    }
//    catch (...)
//    {
//        m_logger.LOG(logger::LogLevel::Error, "Unknown error occured in GridSearch run.");
//    }
//    auto svmTrained = boost::any_cast<phd::svm::OpenCvSvm>(m_gridSearchElement->getOutputPort(0).getValue());
//    return std::make_shared<phd::svm::OpenCvSvm>(svmTrained);
//}
//
//void GridSearchWorkflowOpenCV::connectElements()
//{
//    try
//    {
//        m_validationPath.setValue(m_config.validationDataPath);
//        m_valdiationProviderElement->setInputConnection(0, m_validationPath);
//
//        m_testPath.setValue(m_config.testDataPath);
//        m_testProviderElement->setInputConnection(0, m_testPath);
//
//        m_algorithmConfigDataSource.setValue(&m_algorithmConfig);
//        m_gridSearchElement->setInputConnection(2, m_algorithmConfigDataSource);
//
//        m_outputFilePathSource.setValue(filesystem::Path(m_config.outputFolderPath.string() + m_config.txtLogFilename));
//        m_gridSearchElement->setInputConnection(3, m_outputFilePathSource);
//
//        m_pngFileElement->setInputConnection(0, m_gridSearchElement->getOutputPort(1));
//        m_pngFileElement->setInputConnection(1, m_pngNameSource);
//    }
//    catch (const std::exception& exception)
//    {
//        m_logger.LOG(logger::LogLevel::Error, exception.what());
//    }
//}
//} // namespace genetic
