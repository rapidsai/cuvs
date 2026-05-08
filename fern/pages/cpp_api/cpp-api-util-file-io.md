---
slug: api-reference/cpp-api-util-file-io
---

# File Io

_Source header: `cuvs/util/file_io.hpp`_

## Types

<a id="cuvs-util-fd-streambuf"></a>
### cuvs::util::fd_streambuf

Streambuf that reads from a POSIX file descriptor

```cpp
class fd_streambuf : public std::streambuf { ... };
```

<a id="cuvs-util-fd-istream"></a>
### cuvs::util::fd_istream

Istream that reads from a POSIX file descriptor

```cpp
class fd_istream : public std::istream { ... };
```

<a id="cuvs-util-file-descriptor"></a>
### cuvs::util::file_descriptor

RAII wrapper for POSIX file descriptors

Manages file descriptor lifecycle with automatic cleanup. Non-copyable, move-only.

```cpp
class file_descriptor { ... };
```

<a id="cuvs-util-buffered-ofstream"></a>
### cuvs::util::buffered_ofstream

Buffered output stream wrapper

Wraps an std::ostream with a buffer to improve write performance by reducing the number of system calls. Automatically flushes on destruction. Non-copyable, non-movable.

```cpp
class buffered_ofstream { ... };
```
