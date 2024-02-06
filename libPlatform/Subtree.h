

// #pragma once

// #include <memory>
// #include <boost/property_tree/ptree.hpp>
// #include <boost/property_tree/json_parser.hpp>
// #include <boost/type_index.hpp>

// #include <filesystem>

// #include "SubtreeExceptions.h"


// namespace platform
// {
// class Subtree
// {
//     struct not_vector_tag {};
//     struct vector_tag {};

// public:
//     Subtree() = default;

//     explicit Subtree(const boost::property_tree::ptree& pt);

//     explicit Subtree(const std::string& jsonString);

//     explicit Subtree(const std::filesystem::path& path);

// 	void save(const std::filesystem::path& path) const;

//     template<typename T>
//     void putValue(const std::string& name, const T& value);

//     template<typename T>
//     void putValue(const std::string& name, const std::vector<T>& value);
    
//     void putNode(const std::string& nodeName, const Subtree& node);
//     void putNode(const std::string& nodeName, const std::vector<Subtree>& nodeArray);

//     template<typename T>
//     T getValue(const std::string& propertyName) const;

//     Subtree getNode(const std::string& nodeName) const;

//     std::vector<Subtree> getChildren() const;

//     boost::property_tree::ptree getRoot() const noexcept;

//     bool contains(const std::string& propertyName) const;

//     std::string writeToString() const;

// private:
//     const boost::property_tree::ptree& getChild(const std::string& nodeName) const;

//     template<typename T>
//     T getImplementation(const std::string& propertyName, not_vector_tag, T*) const;

//     template<typename T>
//     std::vector<T> getImplementation(const std::string& propertyName, vector_tag, std::vector<T>*) const;

//     template <typename T, typename _ = void>
//     struct is_vector
//     {
//         using type = not_vector_tag;
//     };

//     template <typename T>
//     struct is_vector< T,
//         typename std::enable_if<std::is_same<T, std::vector< typename T::value_type, typename T::allocator_type >>::value>::type>
//     {
//         using type = vector_tag;
//     };

//     boost::property_tree::ptree m_root;
// };

// template <typename T>
// void Subtree::putValue(const std::string& name, const T& value)
// {
//     if (name.empty())
//     {
//         throw EmptyNameException();
//     }

//     m_root.put<T>(name, value);
// }

// template <typename T>
// void Subtree::putValue(const std::string& name, const std::vector<T>& value)
// {
//     if (name.empty())
//     {
//         throw EmptyNameException();
//     }
//     if (value.empty())
//     {
//         throw EmptyVectorException();
//     }

//     boost::property_tree::ptree children;
//     for (std::size_t i = 0; i < value.size(); i++)
//     {
//         children.push_back(std::make_pair(std::string(), boost::property_tree::ptree().put<T>("", value[i])));
//     }

//     m_root.put_child(name, children);
// }

// template <typename T>
// T Subtree::getValue(const std::string& propertyName) const
// {
//     if (propertyName.empty())
//     {
//         throw EmptyNameException();
//     }
//     try
//     {
//         return getImplementation(propertyName, is_vector<T>::type(), static_cast<T*>(nullptr));
//     }
//     catch (const boost::property_tree::ptree_bad_path&)
//     {
//         throw PropertyNotFoundException(propertyName);
//     }
//     catch (const boost::property_tree::ptree_bad_data&)
//     {
//         throw WrongConversionException(propertyName, boost::typeindex::type_id<T>().pretty_name());
//     }
// }

// template <typename T>
// T Subtree::getImplementation(const std::string& propertyName, not_vector_tag, T*) const
// {
//     return m_root.get<T>(propertyName);
// }

// template <typename T>
// std::vector<T> Subtree::getImplementation(const std::string& propertyName, vector_tag, std::vector<T>*) const
// {
//     std::vector<T> result;
//     auto arrayCointainer = m_root.get_child(propertyName);
//     if (arrayCointainer.empty())
//     {
//         throw WrongConversionException(propertyName, boost::typeindex::type_id<std::vector<T>>().pretty_name());
//     }
//     for (const auto& item : arrayCointainer)
//     {
//         result.emplace_back(item.second.get_value<T>());
//     }
//     return result;
// }
// } // namespace platform

#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "StringUtils.h"
#include "SubtreeExceptions.h"

namespace platform
{
class Subtree
{
public:
    Subtree() = default;

    explicit Subtree(const nlohmann::json& json);

    explicit Subtree(const std::string& jsonString);

    explicit Subtree(const std::filesystem::path& path);

    void save(const std::filesystem::path& path) const;

    template <typename T>
    void putValue(const std::string& name, const T& value);

    template <typename T>
    void putValue(const std::string& name, const std::vector<T>& value);

    void putNode(const std::string& nodeName, const Subtree& node);
    void putNode(const std::string& nodeName, const std::vector<Subtree>& nodeArray);

    template <typename T>
    T getValue(const std::string& propertyName) const;

    Subtree getNode(const std::string& nodeName) const;

    std::vector<Subtree> getChildren() const;

    nlohmann::json getRoot() const noexcept;

    bool contains(const std::string& propertyName) const;

    std::string writeToString() const;

private:
    const nlohmann::json& getChild(const std::string& nodeName) const;

    template <typename T>
    T getImplementation(const std::string& propertyName) const;

    nlohmann::json m_root;
};

template <typename T>
void Subtree::putValue(const std::string& name, const T& value)
{
    if (name.empty())
    {
        throw std::invalid_argument("Empty name");
    }

    //m_root[name] = value;

    using json = nlohmann::json;
    std::vector<std::string> nodeNames = platform::stringUtils::splitString(name, '.');

        // Create the nested structure
    json* currentNode = &m_root;
    for (const auto& nodeName : nodeNames)
    {
        if (!nodeName.empty())
        {
            // If the node doesn't exist, create it as an empty object
            if (!currentNode->contains(nodeName))
            {
                (*currentNode)[nodeName] = json::object();
            }

            // Move to the next level in the hierarchy
            currentNode = &(*currentNode)[nodeName];
        }
    }

    // Set the value for the last node in the hierarchy
    (*currentNode) = value;

}

template <typename T>
void Subtree::putValue(const std::string& name, const std::vector<T>& value)
{
    if (name.empty())
    {
        throw std::invalid_argument("Empty name");
    }
    if (value.empty())
    {
        throw std::invalid_argument("Empty vector");
    }

    //m_root[name] = value;
     // Split the node name by dots and create a nested structure
    using json = nlohmann::json;
    std::vector<std::string> nodeNames = platform::stringUtils::splitString(name, '.');

        // Create the nested structure
    json* currentNode = &m_root;
    for (const auto& nodeName : nodeNames)
    {
        if (!nodeName.empty())
        {
            // If the node doesn't exist, create it as an empty object
            if (!currentNode->contains(nodeName))
            {
                (*currentNode)[nodeName] = json::object();
            }

            // Move to the next level in the hierarchy
            currentNode = &(*currentNode)[nodeName];
        }
    }

    // Set the value for the last node in the hierarchy
    (*currentNode) = value;
}

template <typename T>
T Subtree::getValue(const std::string& propertyName) const
{
    if (propertyName.empty())
    {
        throw std::invalid_argument("Empty name");
    }

    try
    {
        return getImplementation<T>(propertyName);
    }
    catch (const std::out_of_range&)
    {
        throw std::out_of_range("Property not found: " + propertyName);
    }
    catch (const std::invalid_argument&)
    {
        throw std::invalid_argument("Wrong conversion");
    }
}

template <typename T>
T Subtree::getImplementation(const std::string& propertyName) const
{
      std::vector<std::string> nodeNames = platform::stringUtils::splitString(propertyName, '.');
    using json = nlohmann::json;

     // Navigate the nested structure
    const json* currentNode = &m_root;
    for (const auto& nodeName : nodeNames)
    {
        if (!nodeName.empty())
        {
            auto it = currentNode->find(nodeName);

            // If the node doesn't exist, throw an exception
            if (it == currentNode->end())
            {
                throw PropertyNotFoundException(propertyName);
            }

            // Move to the next level in the hierarchy
            currentNode = &it.value();
        }
    }

    // Try to convert the final node to the specified type
    try
    {
        return currentNode->get<T>();
    }
    catch (const std::exception&)
    {
        throw WrongConversionException(propertyName, typeid(T).name());
    }
}
} //namespace platform
