
#include <iostream>
#include "libStrategies/DiskFile.h"
#include "GeneticWorkflowResultLogger.h"
#include "libPlatform/loguru.hpp"

namespace genetic
{
void GeneticWorkflowResultLogger::logToFile(const std::filesystem::path& outputPath)
{
    try
    {
        filesystem::DiskFile resultFile(outputPath, "a");
        for (const auto& logMessage : m_resultEntries)
        {
            resultFile.write(gsl::span<const unsigned char>(reinterpret_cast<const unsigned char*>(logMessage.c_str()), logMessage.length()));
        }
    }
    catch (const std::runtime_error& exception)
    {
        LOG_F(ERROR, "Error: %s", exception.what());
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
