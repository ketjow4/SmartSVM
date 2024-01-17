#include <iomanip>
#include <chrono>
#include <sstream>
#include "libPlatform/TimeUtils.h"
#include "libSvmComponents/SvmVisualization.h"
#include <filesystem> //#include "libFileSystem/FileSystemDefinitions.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
//#include "libFileSystem/FileSystem.h"
#include "WorkflowUtils.h"
#include "SvmWorkflowConfigStruct.h"
#include "libPlatform/TimeUtils.h"

namespace genetic
{


void setVisualizationFilenameAndFormat(svmComponents::imageFormat format,
                                       std::filesystem::path& pngNameSource,
                                       const SvmWokrflowConfiguration& config,
                                       unsigned int generationNumber)
{
    switch (format)
    {
    case svmComponents::imageFormat::png:
        pngNameSource = std::filesystem::path(config.outputFolderPath.string() +
	        //timeUtils::getShortTimestamp() +
            config.visualizationFilename +
            std::to_string(generationNumber) + 
            ".png");
        break;
    default:
        throw svmComponents::UnsupportedImageFormat();
    }
}


void setVisualizationFilenameAndFormat(svmComponents::imageFormat format,
    std::filesystem::path& pngNameSource,
    const SvmWokrflowConfiguration& config)
{
    switch (format)
    {
    case svmComponents::imageFormat::png:
        pngNameSource = std::filesystem::path(config.outputFolderPath.string() +
            //timeUtils::getShortTimestamp() +
            config.visualizationFilename +
            ".png");
        break;
    default:
        throw svmComponents::UnsupportedImageFormat();
    }
}

void setVisualizationFilenameAndFormatWithPrefix(svmComponents::imageFormat format, std::filesystem::path& pngNameSource, const SvmWokrflowConfiguration& config,
                                                 unsigned generationNumber, std::string prefix)
{
	switch (format)
	{
	case svmComponents::imageFormat::png:
		pngNameSource = std::filesystem::path(config.outputFolderPath.string() + prefix +
			//timeUtils::getShortTimestamp() +
			config.visualizationFilename +
			std::to_string(generationNumber) + 
            ".png");
		break;
	default:
		throw svmComponents::UnsupportedImageFormat();
	}
}

std::filesystem::path generateFilenameWithTimestamp(std::filesystem::path filename, std::string prefix , std::filesystem::path outputFolder)
{
     return std::filesystem::path(outputFolder.string() + prefix + timeUtils::getTimestamp() + "__" + filename.string());
}
} // namespace genetic
