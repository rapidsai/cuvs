---
slug: api-reference/cpp-api-util-cutlass-utils
---

# Cutlass Utils

_Source header: `cuvs/util/cutlass_utils.hpp`_

## Types

<a id="cuvs-cutlass-error"></a>
### cuvs::cutlass_error

Exception thrown when a CUTLASS error is encountered.

```cpp
struct cutlass_error : public raft::exception { ... };
```
