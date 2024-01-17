
#include <libStrategies/StrategiesExceptions.h>
#include "SvmTrainingStrategy.h"
#include "SvmExceptions.h"
#include "SvmWorkflowConfigStruct.h"
#include "LocalFileDatasetLoader.h"

namespace strategies
{
std::string SvmTrainingStrategy::getDescription() const
{
    return "This block trains SVM classifier by provided config";
}

std::shared_ptr<phd::svm::ISvm> SvmTrainingStrategy::launch(platform::Subtree& config)
{
  /*  try
    {*/
        auto wokrflowConfig = genetic::SvmWokrflowConfiguration(config);
        genetic::LocalFileDatasetLoader loading(wokrflowConfig.trainingDataPath,
                                                wokrflowConfig.validationDataPath,
                                                wokrflowConfig.testDataPath);

        auto trainingAlgorithm = genetic::SvmAlgorithmFactory::createAlgorightm(config, loading);
        return trainingAlgorithm->run();
    
   /* catch (const genetic::NotImplementedException& exception)
    {
        handleException(exception);
    }
    catch (const genetic::UnknownAlgorithmTypeException& exception)
    {
        handleException(exception);
    }
    catch (const genetic::ErrorInConfigException& exception)
    {
        handleException(exception);
    }
    catch (const platform::PropertyNotFoundException& exception)
    {
        handleException(exception);
    }
    catch (const boost::bad_any_cast& exception)
    {
        handleException(exception);
    }
    return nullptr;*/
}
} // namespace strategies
