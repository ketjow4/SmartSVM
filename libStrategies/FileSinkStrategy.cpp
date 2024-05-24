#include <utility>
#include <vector>
#include <string>

#include "StrategiesExceptions.h"
#include "FileSinkStrategy.h"
#include "DiskFile.h"

namespace strategies
{
std::string FileSinkStrategy::getDescription() const
{
    return "This element provides interface for saving various file into disk space in binary mode";
}

void FileSinkStrategy::launch(gsl::span<unsigned char> data, const std::filesystem::path& filePath)
{
    try
    {
        filesystem::DiskFile file(filePath, "wb");
        file.write(data);
    }
    catch (const std::runtime_error& exception)
	{
		handleException(exception);
	}
    catch(...)
    {
        //m_logger.LOG(logger::LogLevel::Error, "Success error in FileSinkStrategy");
    }
}


void FileSinkStrategy::launch(std::vector<std::pair<std::vector<unsigned char>, std::string>>& data, const std::filesystem::path& filePath, bool postFixFirst)
{
	try
	{
		for (const auto& im : data)
		{
			const auto&[image, name] = im;

			std::filesystem::path newPath;

			if (postFixFirst)
			{
				newPath = filePath.parent_path() / std::filesystem::path(name + "__" + filePath.stem().string() + filePath.extension().string());
			}
			else
			{
				newPath = filePath.parent_path() / std::filesystem::path(filePath.stem().string() + "_" + name + filePath.extension().string());
			}
			

			filesystem::DiskFile file(newPath, "wb");
			file.write(gsl::make_span(image));
		}
	}
	catch (const std::runtime_error& exception)
	{
		handleException(exception);
	}
	catch (...)
	{
		//m_logger.LOG(logger::LogLevel::Error, "Success error in FileSinkStrategy");
	}
}

void FileSinkStrategy::handleException(const std::runtime_error& /*exception*/)
{
   //m_logger.LOG(logger::LogLevel::Error, exception.what());
}
} // namespace strategies