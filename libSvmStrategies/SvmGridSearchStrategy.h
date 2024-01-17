//
//#pragma once
//
//#include <libSvmComponents/GridSearchImplementation.h>
//
//namespace svmStrategies
//{
//class SvmGridSearchStrategy
//{
//public:
//    SvmGridSearchStrategy();
//
//  
//    std::string getDescription() const;
//    void launch(framework::InputReader&& inputReader, framework::OutputWriter&& outputWriter) override;
//
//private:
//    /*bool isInitialized(framework::InputReader inputReader) const;*/
//
//    std::unique_ptr<svmComponents::GridSearchImplementation> m_gridSearchAlgorithm;
//};
//} // namespace svmStrategies