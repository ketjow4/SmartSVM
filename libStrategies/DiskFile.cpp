

#include <memory>

#include "DiskFile.h"

namespace filesystem
{
DiskFile::DiskFile(const std::filesystem::path& path, const char* mode)
    : m_file{
        [&]()
        {
            FILE* ptr = nullptr;

            if (fopen_s(&ptr, path.string().c_str(), mode) != 0)
            {
                throw std::exception(path.string().c_str());
            }

            return ptr;
        }()
    }
    , m_path{path}
{
}

DiskFile::~DiskFile()
{
    fclose(m_file);
}

gsl::span<uint8_t> DiskFile::read(gsl::span<uint8_t> buffer)
{
    const auto byteCount = fread(buffer.data(), sizeof(uint8_t), buffer.size(), m_file);

    if (ferror(m_file))
    {
        //throw ReadFailedException(buffer.size());
    }

    return buffer.subspan(0, byteCount);
}

gsl::span<const uint8_t> DiskFile::write(gsl::span<const uint8_t> buffer)
{
    const auto byteCount = fwrite(buffer.data(), sizeof(uint8_t), buffer.size(), m_file);

    if (ferror(m_file))
    {
        //throw WriteFailedException(buffer.size());
    }

    return buffer.subspan(0, byteCount);
}

bool DiskFile::seek(uintmax_t offset, std::ios_base::seekdir way)
{
    const auto ret = fseek(m_file, static_cast<long>(offset), way);

    if (ferror(m_file))
    {
        //throw SeekFailedException(offset, way);
    }

    return ret == 0;
}

std::streampos DiskFile::tell() const
{
    const auto ret = ftell(m_file);

    if (ferror(m_file))
    {
        //throw TellFailedException();
    }

    return ret;
}

uintmax_t DiskFile::size() const
{
    return file_size(m_path);
}
} // namespace filesystem
