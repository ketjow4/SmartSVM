
#pragma once

#include <unordered_map>
#include "ISvm.h"
//#include "libSvmComponents/IGroupPropagation.h"

namespace phd { namespace svm
{
enum class SvmImplementationType
{
    Unknown,
    OpenCvSvm,
    LibSvm
};

class SvmFactory
{
public:
    //static std::unique_ptr<ISvm> create(SvmImplementationType implementationType, std::shared_ptr<svmComponents::IGroupPropagation> propagation);
    static std::unique_ptr<ISvm> create(SvmImplementationType implementationType);

    static SvmImplementationType implementationTypeFromString(const std::string& implementationName);

private:
    const static std::unordered_map<std::string, SvmImplementationType> m_translations;
};
}} // namespace phd::svm
