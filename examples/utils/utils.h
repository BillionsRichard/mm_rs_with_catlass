#include <climits>
#include <fstream>
#include <iostream>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        ERROR_LOG("Open file failed. path = %s, error: %s", filePath.c_str(), strerror(errno));
        return false;
    }

    // lock
    if (flock(fd, LOCK_EX) == -1) {
        ERROR_LOG("Failed to acquire lock for file: %s, error: %s", filePath.c_str(), strerror(errno));
        close(fd);
        return false;
    }

    size_t written = 0;
    size_t chunkSize = 4 * 1024 * 1024; // 4 MB
    const char *data = static_cast<const char *>(buffer);

    while (written < size) {
        size_t remaining = size - written;
        size_t toWrite = (remaining < chunkSize) ? remaining : chunkSize;

        if (lseek(fd, offset + written, SEEK_SET) == -1) {
            ERROR_LOG("Failed to seek in file: %s, error: %s", filePath.c_str(), strerror(errno));
            flock(fd, LOCK_UN);
            close(fd);
            return false;
        }

        ssize_t bytesWritten = write(fd, data + written, toWrite);
        if (bytesWritten != static_cast<ssize_t>(toWrite)) {
            ERROR_LOG("Failed to write to file: %s, written %zd bytes out of %zu, error: %s",
                      filePath.c_str(), bytesWritten, toWrite, strerror(errno));
            flock(fd, LOCK_UN);
            close(fd);
            return false;
        }

        written += bytesWritten;
    }

    // unlock
    flock(fd, LOCK_UN);
    close(fd);
    return true;
}
