# ABI Stability

ABI stands for **Application Binary Interface**. It defines how compiled
programs interact with a shared library at runtime.

For cuVS, ABI stability means that an application built against one supported
version of the **cuVS C library** can run with a compatible newer cuVS runtime
without being rebuilt.

ABI stability is different from API stability:

| Concept | Meaning |
|---|---|
| API stability | Source code continues to compile. |
| ABI stability | Already-compiled binaries continue to run. |

## Compatibility Rule

A cuVS runtime is ABI-compatible with an application when both of the following
are true:

1. The runtime has the same ABI major version as the version used at build time.
2. The runtime cuVS version is the same or newer than the version used at build
   time.

For example, an application built against cuVS 26.04 can use a cuVS 26.06
runtime when both versions use the same ABI major version. An older runtime or a
runtime with a different ABI major version is not considered compatible.

## Scope

The ABI stability guarantee applies to the public **C ABI**:

- Public C headers installed under `include/cuvs/`
- Functions, enums, and structs declared in `.h` headers
- Public symbols exposed through `extern "C"`

The guarantee does not generally apply to internal implementation details or to
the C++ implementation ABI.

## Public Structs

Public structs require care because compiled applications may depend on their
binary layout. Prefer create and destroy functions for objects that own memory
or expose implementation details.

Recommended pattern:

```c
cuvsHnswIndex_t index;
cuvsHnswIndexCreate(&index);

cuvsHnswIndexParams_t params;
cuvsHnswIndexParamsCreate(&params);

cuvsHnswFromCagra(res, params, cagra_index, index);
```

Avoid requiring users to allocate or initialize structs directly when future
fields may need to be added.

## C++ ABI

Guaranteeing ABI stability for C++ is difficult because the binary interface can
change when templates, inline functions, namespaces, class layouts, standard
library types, compiler versions, or CUDA compilation details change.

For this reason, cuVS treats the C API as the stable ABI boundary. Language
bindings and downstream integrations should call the C API rather than depending
on C++ symbols directly when binary compatibility across releases is required.
