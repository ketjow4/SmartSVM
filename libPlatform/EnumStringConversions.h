#pragma once

#include <string>
#include <unordered_map>
#include "PlatformException.h"

namespace platform
{
class UnknownEnumType final : public PlatformException
{
public:
    explicit UnknownEnumType(const std::string& name, const std::string& enumName);
    explicit UnknownEnumType(const std::string& enumName);
};

inline UnknownEnumType::UnknownEnumType(const std::string& name, const std::string& enumName)
    : PlatformException("Enum type:" + name + " is unknown name in:" + enumName)
{
}

inline UnknownEnumType::UnknownEnumType(const std::string& enumName)
    : PlatformException("Unknown enum value for: " + enumName)
{
}

template <class EnumeratorT>
class UnknownTranslation final : public platform::PlatformException
{
public:
    explicit UnknownTranslation(const std::string& enumName, EnumeratorT enumValue);
};

template <class EnumeratorT>
UnknownTranslation<EnumeratorT>::UnknownTranslation(const std::string& enumName, EnumeratorT enumValue)
    : PlatformException("Cannot find translation for enum class " + enumName + 
                        " value of enum as int: " + std::to_string(static_cast<int>(enumValue)))
{
}

template <class EnumeratorT>
EnumeratorT stringToEnum(const std::string& name, std::unordered_map<std::string, EnumeratorT> translations)
{
    auto iterator = translations.find(name);
    if (iterator != translations.end())
    {
        return iterator->second;
    }
    throw UnknownEnumType(name, typeid(EnumeratorT).name());
}

template <class EnumeratorT>
EnumeratorT stringToEnumWithDefault(const std::string& name,
                                   std::unordered_map<std::string, EnumeratorT> translations,
                                    EnumeratorT defaultValue)
{
    auto iterator = translations.find(name);
    if (iterator != translations.end())
    {
        return iterator->second;
    }
    return defaultValue;
}


template <class EnumeratorT>
std::string enumToString(EnumeratorT enumValue, std::unordered_map<EnumeratorT, std::string> translations)
{
    auto iterator = translations.find(enumValue);
    if (iterator != translations.end())
    {
        return iterator->second;
    }
    throw UnknownTranslation<EnumeratorT>(typeid(EnumeratorT).name(), enumValue);
}
} // namespace platform
