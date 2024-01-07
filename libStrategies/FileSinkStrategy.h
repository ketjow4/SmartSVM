
#pragma once

#include <filesystem>
#include <gsl/span>

namespace strategies
{
class FileSinkStrategy 
{
public:
    std::string getDescription() const;
    void launch(gsl::span<unsigned char> data, const std::filesystem::path& filePath);

	void launch(std::vector<std::pair<std::vector<unsigned char>, std::string>>& data, const std::filesystem::path& filePath, bool postFixFirst = false);

private:
    void handleException(const std::exception& exception);

  
};
} // namespace strategies