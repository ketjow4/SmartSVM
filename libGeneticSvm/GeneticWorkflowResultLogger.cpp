
#include <iostream>
#include "libStrategies/DiskFile.h"
#include "GeneticWorkflowResultLogger.h"

namespace genetic
{
void GeneticWorkflowResultLogger::logToFile(const std::filesystem::path& outputPath)
{
    //logger::LogFrontend logger;
    try
    {
        filesystem::DiskFile resultFile(outputPath, "a");
        for (const auto& logMessage : m_resultEntries)
        {
            resultFile.write(gsl::span<const unsigned char>(reinterpret_cast<const unsigned char*>(logMessage.c_str()), logMessage.length()));
        }
    }
    catch (const std::exception& /*exception*/)
    {
        //logger.LOG(logger::LogLevel::Error, exception.what());
    }
}

void GeneticWorkflowResultLogger::logToConsole()
{
    for (const auto& logMessage : m_resultEntries)
    {
        std::cout << logMessage;
    }
}

void GeneticWorkflowResultLogger::setEntries(std::vector<std::string>& entires)
{
    m_resultEntries = entires;
}
} // namespace genetic
