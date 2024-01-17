//
//#pragma once
//
//#include <libFramework/Element.h>
//#include <libFramework/Scheduler.h>
//#include <libSvmComponents/SvmConfigStructures.h>
//#include "libGeneticSvm/ISvmAlgorithm.h"
//#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
//#include "IDatasetLoader.h"
//
//namespace genetic
//{
///*
// * @wdudzik This code is deprecated
// */
//class GridSearchWorkflowOpenCV : public ISvmAlgorithm
//{
//public:
//    explicit GridSearchWorkflowOpenCV(const SvmWokrflowConfiguration& config,
//                                      svmComponents::GridSearchConfiguration&& algorithmConfig,
//                                      IDatasetLoader& workflow);
//
//    std::shared_ptr<phd::svm::ISvm> run() override;
//
//private:
//    void moveStrategiesToElements();
//    void connectElements();
//    void connectDataToSecondScheduler();
//    bool isNotLastIteration(unsigned int iteration) const;
//    void runGridSearch();
//
//    //elements
//    std::shared_ptr<framework::Element> m_valdiationProviderElement;
//    std::shared_ptr<framework::Element> m_testProviderElement;
//    std::shared_ptr<framework::Element> m_gridSearchElement;
//    std::shared_ptr<framework::Element> m_pngFileElement;
//
//    //dataSources
//    framework::DataSource m_algorithmConfigDataSource;
//    framework::DataSource m_validationPath;
//    framework::DataSource m_testPath;
//    framework::DataSource m_validationSource;
//    framework::DataSource m_testSource;
//    framework::DataSource m_pngNameSource;
//    framework::DataSource m_outputFilePathSource;
//
//    IDatasetLoader& m_workflow;
//    svmComponents::GridSearchConfiguration m_algorithmConfig;
//    const SvmWokrflowConfiguration m_config;
//    logger::LogFrontend m_logger;
//};
//} // namespace genetic
