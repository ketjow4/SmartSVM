#pragma once

#include "libPlatform/Subtree.h"
namespace platform
{
// template <class T>
// class ConfigParser : public T
// {
// public:
//     using T::read;
//     using T::dump;
// };

// class JSON
// {
// public:
//     Subtree read(const filesystem::Path& pathToFile) const
//     {
//         //if(boost::filesystem::exists(pathToFile))
//         //{
//         //    throw FileNotFoundException(pathToFile.string());
//         //}

//         boost::property_tree::ptree root;
//         try
//         {
//             boost::property_tree::read_json(pathToFile.string(), root);
//         }
//         catch (const boost::property_tree::json_parser_error& exception)
//         {
//             throw JsonParserException(exception.what());
//         }

//         return Subtree(root);
//     }

//     void dump(const filesystem::Path& pathToFile, const Subtree& in, bool pretty = true) const
//     {
//         if (pathToFile.empty())
//         {
//             throw EmptyNameException();
//         }

//         try
//         {
//             boost::property_tree::write_json(pathToFile.string(), in.getRoot(), std::locale(), pretty);
//         }
//         catch (const boost::property_tree::json_parser_error& exception)
//         {
//             throw JsonParserException(exception.what());
//         }
//     }
// };
} // namespace platform
