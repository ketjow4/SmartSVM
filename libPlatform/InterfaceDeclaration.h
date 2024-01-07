

#pragma once

#ifdef LIBLOGGER_EXPORTS
#define LIBLOGGER_API __declspec(dllexport)
#else
#define LIBLOGGER_API __declspec(dllimport)
#endif