
#include "SvmFactory.h"
//#include "OpenCvSvm.h"
#include "libSvmImplementation.h"


namespace phd { namespace svm
{
const std::unordered_map<std::string, SvmImplementationType> SvmFactory::m_translations =
{
    { "OpenCvSvm", SvmImplementationType::OpenCvSvm },
    { "LibSvm", SvmImplementationType::LibSvm}
};

//std::unique_ptr<ISvm> SvmFactory::create(SvmImplementationType implementationType, std::shared_ptr<svmComponents::IGroupPropagation> propagation)
std::unique_ptr<ISvm> SvmFactory::create(SvmImplementationType implementationType)
{
    switch (implementationType)
    {
    // case SvmImplementationType::OpenCvSvm:
    // {
    //     return std::make_unique<OpenCvSvm>();
    // }
    case SvmImplementationType::LibSvm:
    {
        //return std::make_unique<libSvmImplementation>(propagation);
        return std::make_unique<libSvmImplementation>();
    }
    default: throw UnknownEnumType(typeid(SvmImplementationType).name());
    }
}

SvmImplementationType SvmFactory::implementationTypeFromString(const std::string& implementationName)
{
    const auto iterator = m_translations.find(implementationName);
    if (iterator != m_translations.end())
    {
        return iterator->second;
    }
    throw UnknownEnumType(implementationName, typeid(SvmImplementationType).name());
}
}} // namespace phd::svm
