

#pragma once

#include <gsl/span>
#include <filesystem>

namespace filesystem
{
class DiskFile
{
public:
    explicit DiskFile(const std::filesystem::path& path, const char* mode);

    ~DiskFile();

    gsl::span<std::uint8_t> read(gsl::span<std::uint8_t> buffer);

    gsl::span<const std::uint8_t> write(gsl::span<const std::uint8_t> buffer);

    bool seek(std::uintmax_t offset, std::ios_base::seekdir way);

    std::streampos tell() const;

    std::uintmax_t size() const;

private:
    FILE* m_file;
    std::filesystem::path m_path;
};

} // namespace filesystem
