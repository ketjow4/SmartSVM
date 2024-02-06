

// #include "Subtree.h"

// namespace platform
// {
// Subtree::Subtree(const boost::property_tree::ptree& pt)
//     : m_root(pt)
// {
// }

// Subtree::Subtree(const std::string& jsonString)
// {
//     try
//     {
//         std::stringstream  iss;
//         iss << jsonString;
//         boost::property_tree::read_json(iss, m_root);
//     }
//     catch (const std::exception& e)
//     {
//         throw RootCreationFailedException(e.what());
//     }
//     catch (...)
//     {
//         throw UnhandledBoostException();
//     }
// }

// void Subtree::save(const std::filesystem::path& path) const
// {
// 	const auto pretty = true;
// 	boost::property_tree::write_json(path.string(), m_root, std::locale(), pretty);
// }

// Subtree::Subtree(const std::filesystem::path& path)
// {
//     try
//     {
//         boost::property_tree::read_json(path.string(), m_root);
//     }
//     catch (const std::exception& e)
//     {
//         throw RootCreationFailedException(e.what());
//     }
//     catch (...)
//     {
//         throw UnhandledBoostException();
//     }
// }

// void Subtree::putNode(const std::string& nodeName, const Subtree& node)
// {
//     if (nodeName.empty())
//     {
//         throw EmptyNameException();
//     }
//     m_root.put_child(nodeName, node.m_root);
// }

// void Subtree::putNode(const std::string& nodeName, const std::vector<Subtree>& nodeArray)
// {
//     if (nodeName.empty())
//     {
//         throw EmptyNameException();
//     }

//     boost::property_tree::ptree children;
//     for (auto& item : nodeArray)
//     {
//         children.push_back(std::make_pair(std::string(), item.m_root));
//     }

//     m_root.put_child(nodeName, children);
// }

// const boost::property_tree::ptree& Subtree::getChild(const std::string& nodeName) const
// {
//     try
//     {
//         return m_root.get_child(nodeName);
//     }
//     catch (const boost::property_tree::ptree_bad_path&)
//     {
//         throw ChildNotFoundException(nodeName);
//     }
// }

// Subtree Subtree::getNode(const std::string& nodeName) const
// {
//     auto child = getChild(nodeName);
//     return Subtree(child);
// }

// std::vector<Subtree> Subtree::getChildren() const
// {
//     std::vector<Subtree> children;
//     auto item = getChild(std::string());

//     children.reserve(item.size());
//     for (auto& i : item)
//     {
//         children.emplace_back(Subtree(i.second));
//     }
//     return children;
// }

// boost::property_tree::ptree Subtree::getRoot() const noexcept
// {
//     return m_root;
// }

// bool Subtree::contains(const std::string& propertyName) const
// {
//     auto child = m_root.get_child_optional(propertyName);
//     return !!child;
// }

// std::string Subtree::writeToString() const
// {
//     std::stringstream ss;
//     write_json(ss, m_root);

//     return  ss.str();
// }
// } // namespace platform


#include <fstream>
#include "Subtree.h"

namespace platform
{
Subtree::Subtree(const nlohmann::json& json)
    : m_root(json)
{
}

Subtree::Subtree(const std::string& jsonString)
{
    try
    {
        m_root = nlohmann::json::parse(jsonString);
    }
    catch (const std::exception& e)
    {
        throw RootCreationFailedException(e.what());
    }
    catch (...)
    {
        throw UnhandledBoostException();
    }
}

void Subtree::save(const std::filesystem::path& path) const
{
    const auto pretty = true;
    std::ofstream file(path);
    file << std::setw(4) << m_root << std::endl;
}

Subtree::Subtree(const std::filesystem::path& path)
{
    try
    {
        std::ifstream file(path);
        m_root = nlohmann::json::parse(file);
    }
    catch (const std::exception& e)
    {
        throw RootCreationFailedException(e.what());
    }
    catch (...)
    {
        throw UnhandledBoostException();
    }
}

void Subtree::putNode(const std::string& nodeName, const Subtree& node)
{
    // Split the node name by dots and navigate the nested structure
    std::vector<std::string> nodeNames = platform::stringUtils::splitString(nodeName, '.');

    // Navigate the nested structure
    nlohmann::json* currentNode = &m_root;
    for (const auto& name : nodeNames)
    {
        if (!name.empty())
        {
            auto it = currentNode->find(name);

            // If the node doesn't exist, create it
            if (it == currentNode->end())
            {
                it = currentNode->emplace(name, nlohmann::json::object()).first;
            }

            // Move to the next level in the hierarchy
            currentNode = &it.value();
        }
    }

    // Assign the value to the final node
    *currentNode = node.getRoot();
}

void Subtree::putNode(const std::string& nodeName, const std::vector<Subtree>& nodeArray)
{
    if (nodeName.empty())
    {
        throw EmptyNameException();
    }

    for (size_t i = 0; i < nodeArray.size(); ++i)
    {
        m_root[nodeName][i] = nodeArray[i].m_root;
    }
}

const nlohmann::json& Subtree::getChild(const std::string& nodeName) const
{
    try
    {
    std::vector<std::string> nodeNames = platform::stringUtils::splitString(nodeName, '.');

    // Navigate the nested structure
    const nlohmann::json* currentNode = &m_root;
    for (const auto& name : nodeNames)
    {
        if (!name.empty())
        {
            auto it = currentNode->find(name);

            // If the node doesn't exist, throw an exception
            if (it == currentNode->end())
            {
                throw ChildNotFoundException(name);
            }

            // Move to the next level in the hierarchy
            currentNode = &it.value();
        }
    }

    return *currentNode;
    }
    catch (const std::exception&)
    {
        throw ChildNotFoundException(nodeName);
    }
}

Subtree Subtree::getNode(const std::string& nodeName) const
{
    auto child = getChild(nodeName);
    return Subtree(child);
}

std::vector<Subtree> Subtree::getChildren() const
{
    std::vector<Subtree> children;
    for (auto& item : m_root.items())
    {
        children.emplace_back(Subtree(item.value()));
    }
    return children;
}

nlohmann::json Subtree::getRoot() const noexcept
{
    return m_root;
}

bool Subtree::contains(const std::string& propertyName) const
{
    return m_root.find(propertyName) != m_root.end();
}

std::string Subtree::writeToString() const
{
    return m_root.dump(4);
}
} // namespace platform