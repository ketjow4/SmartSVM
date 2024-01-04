#include <thread>
#include <sstream>
#include <string_view>
#include <Windows.h>
#include "SubprocessExceptions.h"
#include "Subprocess.h"

namespace platform::subprocess
{
static std::string GetLastErrorAsString()
{
    const auto errorMessageId = ::GetLastError();
    if (errorMessageId == 0)
    {
        return std::string(); //No error message has been recorded
    }

    LPSTR messageBuffer = nullptr;
    const size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                       nullptr, errorMessageId, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, nullptr);

    std::string message(messageBuffer, size);

    LocalFree(messageBuffer);

    return message;
}

static BOOL IsProcessRunning(HANDLE process)
{
    return WaitForSingleObject(process, 0) != WAIT_OBJECT_0;
}

static PROCESS_INFORMATION CreateChildProcess(const std::string& command, HANDLE childStdOutWrite)
{
    PROCESS_INFORMATION processInformation;
    STARTUPINFO startInfo;

    ZeroMemory(&processInformation, sizeof(PROCESS_INFORMATION));
    ZeroMemory(&startInfo, sizeof(STARTUPINFO));

    startInfo.cb = sizeof(STARTUPINFO);
    startInfo.hStdError = childStdOutWrite;
    startInfo.hStdOutput = childStdOutWrite;
    startInfo.dwFlags |= STARTF_USESTDHANDLES;

    const auto success = CreateProcess(nullptr,
                                       (LPSTR)command.c_str(),
                                       nullptr,
                                       nullptr,
                                       TRUE, //@wdudzik handles are inherited 
                                       NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW,
                                       nullptr,
                                       nullptr,
                                       &startInfo,
                                       &processInformation);

    if (success)
    {
        return processInformation;
    }
    const auto errorMessage = GetLastErrorAsString();
    throw CouldNotCreateProcess(errorMessage, command);
}

static std::string ReadFromPipe(HANDLE childStdOutRead, PROCESS_INFORMATION processInformation)
{
    constexpr auto bufferSize = 4096u;
    std::stringstream ss;
    std::string buffer(bufferSize, 0);    

    while (true)
    {
        DWORD bytesAvailable = 0;
        if (!PeekNamedPipe(childStdOutRead, nullptr, 0, nullptr, &bytesAvailable, nullptr))
        {
            break;
        }
        if (bytesAvailable)
        {
            DWORD numberOfBytesReaded = 0u;
            const auto success = ReadFile(childStdOutRead, buffer.data(), static_cast<DWORD>(buffer.size()), &numberOfBytesReaded, nullptr);
            if (!success || numberOfBytesReaded == 0)
            {
                //@wdudzik this can only happen when handle is closed but if we enter to ReadFile it couldn't be closed
                break;
            }
            ss << std::string_view(buffer.data(), numberOfBytesReaded);
        }
        if(!IsProcessRunning(processInformation.hProcess) && bytesAvailable == 0)
        {
            CloseHandle(childStdOutRead);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); //@wdudzik no need to ask more frequently for input
    }
    return ss.str();
}

std::pair<std::string, DWORD> launchWithPipe(const std::string& command)
{
    SECURITY_ATTRIBUTES securityAttributes;
    securityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
    securityAttributes.bInheritHandle = TRUE;
    securityAttributes.lpSecurityDescriptor = nullptr;

    HANDLE childStdOutRead = nullptr;
    HANDLE childStdOutWrite = nullptr;

    //@wdudzik Create a pipe for the child process's STDOUT. 
    if (!CreatePipe(&childStdOutRead, &childStdOutWrite, &securityAttributes, 0))
    {
        throw ErrorOnCreatingPipes(GetLastErrorAsString());
    }

    if (!SetHandleInformation(childStdOutRead, HANDLE_FLAG_INHERIT, 0))
    {
        throw ErrorOnCreatingPipes(GetLastErrorAsString());
    }

    const auto processInformation = CreateChildProcess(command, childStdOutWrite);
    auto output = ReadFromPipe(childStdOutRead, processInformation);

    DWORD exitCode;
    const auto result = GetExitCodeProcess(processInformation.hProcess, &exitCode);

    if(!result)
    {
        throw CouldNotRetriveExitCode(GetLastErrorAsString());
    }

    CloseHandle(processInformation.hProcess);
    CloseHandle(processInformation.hThread);
    CloseHandle(childStdOutWrite);

    return { output, exitCode };
}

int launch(const std::string& command)
{
    STARTUPINFO startupInformation;
    PROCESS_INFORMATION processInformation;

    ZeroMemory(&startupInformation, sizeof(startupInformation));
    startupInformation.cb = sizeof(startupInformation);
    ZeroMemory(&processInformation, sizeof(processInformation));

    auto result = CreateProcess(nullptr,
                                (LPSTR)command.c_str(), // @rlp: Probably the best option here
                                nullptr,
                                nullptr,
                                FALSE,
                                NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW,
                                nullptr,
                                nullptr,
                                &startupInformation,
                                &processInformation);

    if (!result)
    {
        return GetLastError();
    }

    WaitForSingleObject(processInformation.hProcess, INFINITE);

    DWORD exitCode;
    result = GetExitCodeProcess(processInformation.hProcess, &exitCode);

    CloseHandle(processInformation.hProcess);
    CloseHandle(processInformation.hThread);

    return result;
}
} // namespace platform::subprocess
