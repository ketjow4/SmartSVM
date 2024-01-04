

#include "Subtree.h"

namespace platform
{
Subtree::Subtree(const boost::property_tree::ptree& pt)
    : m_root(pt)
{
}

Subtree::Subtree(const std::string& jsonString)
{
    try
    {
        std::stringstream  iss;
        iss << jsonString;
        boost::property_tree::read_json(iss, m_root);
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
	boost::property_tree::write_json(path.string(), m_root, std::locale(), pretty);
}

Subtree::Subtree(const std::filesystem::path& path)
{
    try
    {
        boost::property_tree::read_json(path.string(), m_root);
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
    if (nodeName.empty())
    {
        throw EmptyNameException();
    }
    m_root.put_child(nodeName, node.m_root);
}

void Subtree::putNode(const std::string& nodeName, const std::vector<Subtree>& nodeArray)
{
    if (nodeName.empty())
    {
        throw EmptyNameException();
    }

    boost::property_tree::ptree children;
    for (auto& item : nodeArray)
    {
        children.push_back(std::make_pair(std::string(), item.m_root));
    }

    m_root.put_child(nodeName, children);
}

const boost::property_tree::ptree& Subtree::getChild(const std::string& nodeName) const
{
    try
    {
        return m_root.get_child(nodeName);
    }
    catch (const boost::property_tree::ptree_bad_path&)
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
    auto item = getChild(std::string());

    children.reserve(item.size());
    for (auto& i : item)
    {
        children.emplace_back(Subtree(i.second));
    }
    return children;
}

boost::property_tree::ptree Subtree::getRoot() const noexcept
{
    return m_root;
}

bool Subtree::contains(const std::string& propertyName) const
{
    auto child = m_root.get_child_optional(propertyName);
    return !!child;
}

std::string Subtree::writeToString() const
{
    std::stringstream ss;
    write_json(ss, m_root);

    return  ss.str();
}
} // namespace platform
