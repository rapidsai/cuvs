---
slug: api-reference/c-api-core-c-api
---

# C API

_Source header: `cuvs/core/c_api.h`_

## cuVS Error Messages

<a id="cuvserror-t"></a>
### cuvsError_t

An enum denoting error statuses for function calls

```c
typedef enum { ... } cuvsError_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_ERROR` | `0` |
| `CUVS_SUCCESS` | `1` |

<a id="cuvsgetlasterrortext"></a>
### cuvsGetLastErrorText

Returns a string describing the last seen error on this thread, or

```c
CUVS_EXPORT const char* cuvsGetLastErrorText();
```

NULL if the last function succeeded.

**Returns**

`CUVS_EXPORT const char*`

<a id="cuvssetlasterrortext"></a>
### cuvsSetLastErrorText

Sets a string describing an error seen on the thread. Passing NULL

```c
CUVS_EXPORT void cuvsSetLastErrorText(const char* error);
```

clears any previously seen error message.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `error` |  | `const char*` |  |

**Returns**

`CUVS_EXPORT void`

## cuVS Logging

<a id="cuvsloglevel-t"></a>
### cuvsLogLevel_t

An enum denoting log levels

```c
typedef enum { ... } cuvsLogLevel_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_LOG_LEVEL_TRACE` | `0` |
| `CUVS_LOG_LEVEL_DEBUG` | `1` |
| `CUVS_LOG_LEVEL_INFO` | `2` |
| `CUVS_LOG_LEVEL_WARN` | `3` |
| `CUVS_LOG_LEVEL_ERROR` | `4` |
| `CUVS_LOG_LEVEL_CRITICAL` | `5` |
| `CUVS_LOG_LEVEL_OFF` | `6` |

<a id="cuvsgetloglevel"></a>
### cuvsGetLogLevel

Returns the current log level

```c
CUVS_EXPORT cuvsLogLevel_t cuvsGetLogLevel();
```

**Returns**

[`CUVS_EXPORT cuvsLogLevel_t`](/api-reference/c-api-core-c-api#cuvsloglevel-t)

<a id="cuvssetloglevel"></a>
### cuvsSetLogLevel

Sets the log level

```c
CUVS_EXPORT void cuvsSetLogLevel(cuvsLogLevel_t);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`cuvsLogLevel_t`](/api-reference/c-api-core-c-api#cuvsloglevel-t) |  |

**Returns**

`CUVS_EXPORT void`

## cuVS Resources Handle

<a id="cuvsresources-t"></a>
### cuvsResources_t

An opaque C handle for C++ type `raft::resources`

```c
typedef uintptr_t cuvsResources_t;
```

<a id="cuvsresourcescreate"></a>
### cuvsResourcesCreate

Create an Initialized opaque C handle for C++ type `raft::resources`

```c
CUVS_EXPORT cuvsError_t cuvsResourcesCreate(cuvsResources_t* res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t*`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsresourcesdestroy"></a>
### cuvsResourcesDestroy

Destroy and de-allocate opaque C handle for C++ type `raft::resources`

```c
CUVS_EXPORT cuvsError_t cuvsResourcesDestroy(cuvsResources_t res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsstreamset"></a>
### cuvsStreamSet

Set cudaStream_t on cuvsResources_t to queue CUDA kernels on APIs

```c
CUVS_EXPORT cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream);
```

that accept a cuvsResources_t handle

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `stream` | in | `cudaStream_t` | cudaStream_t stream to queue CUDA kernels |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsstreamget"></a>
### cuvsStreamGet

Get the cudaStream_t from a cuvsResources_t

```c
CUVS_EXPORT cuvsError_t cuvsStreamGet(cuvsResources_t res, cudaStream_t* stream);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `stream` | out | `cudaStream_t*` | cudaStream_t stream to queue CUDA kernels |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsstreamsync"></a>
### cuvsStreamSync

Syncs the current CUDA stream on the resources object

```c
CUVS_EXPORT cuvsError_t cuvsStreamSync(cuvsResources_t res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsdeviceidget"></a>
### cuvsDeviceIdGet

Get the id of the device associated with this cuvsResources_t

```c
CUVS_EXPORT cuvsError_t cuvsDeviceIdGet(cuvsResources_t res, int* device_id);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `device_id` | out | `int*` | int the id of the device associated with res |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuresourcescreate"></a>
### cuvsMultiGpuResourcesCreate

Create an Initialized opaque C handle for C++ type `raft::device_resources_snmg`

```c
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesCreate(cuvsResources_t* res);
```

for multi-GPU operations

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t*`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuresourcescreatewithdeviceids"></a>
### cuvsMultiGpuResourcesCreateWithDeviceIds

Create an Initialized opaque C handle for C++ type `raft::device_resources_snmg`

```c
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesCreateWithDeviceIds(cuvsResources_t* res,
DLManagedTensor* device_ids);
```

for multi-GPU operations with specific device IDs

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t*`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `device_ids` | in | `DLManagedTensor*` | DLManagedTensor* containing device IDs to use |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuresourcesdestroy"></a>
### cuvsMultiGpuResourcesDestroy

Destroy and de-allocate opaque C handle for C++ type `raft::device_resources_snmg`

```c
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesDestroy(cuvsResources_t res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuresourcessetmemorypool"></a>
### cuvsMultiGpuResourcesSetMemoryPool

Set a memory pool on all devices managed by the multi-GPU resources

```c
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesSetMemoryPool(cuvsResources_t res, int percent_of_free_memory);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle for multi-GPU resources |
| `percent_of_free_memory` | in | `int` | Percent of free memory to allocate for the pool |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## cuVS Memory Allocation

<a id="cuvsrmmalloc"></a>
### cuvsRMMAlloc

Allocates device memory using RMM

```c
CUVS_EXPORT cuvsError_t cuvsRMMAlloc(cuvsResources_t res, void** ptr, size_t bytes);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `ptr` | out | `void**` | Pointer to allocated device memory |
| `bytes` | in | `size_t` | Size in bytes to allocate |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrmmfree"></a>
### cuvsRMMFree

Deallocates device memory using RMM

```c
CUVS_EXPORT cuvsError_t cuvsRMMFree(cuvsResources_t res, void* ptr, size_t bytes);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `ptr` | in | `void*` | Pointer to allocated device memory to free |
| `bytes` | in | `size_t` | Size in bytes to allocate |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrmmpoolmemoryresourceenable"></a>
### cuvsRMMPoolMemoryResourceEnable

Switches the working memory resource to use the RMM pool memory resource, which will

```c
CUVS_EXPORT cuvsError_t cuvsRMMPoolMemoryResourceEnable(int initial_pool_size_percent,
int max_pool_size_percent,
bool managed);
```

bypass unnecessary synchronizations by allocating a chunk of device memory up front and carving that up for temporary memory allocations within algorithms. Be aware that this function will change the memory resource for the whole process and the new memory resource will be used until explicitly changed.

available memory available memory

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `initial_pool_size_percent` | in | `int` | The initial pool size as a percentage of the total |
| `max_pool_size_percent` | in | `int` | The maximum pool size as a percentage of the total |
| `managed` | in | `bool` | Whether to use a managed memory resource as upstream resource or not |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrmmmemoryresourcereset"></a>
### cuvsRMMMemoryResourceReset

Resets the memory resource to use the default memory resource (cuda_memory_resource)

```c
CUVS_EXPORT cuvsError_t cuvsRMMMemoryResourceReset();
```

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrmmhostalloc"></a>
### cuvsRMMHostAlloc

Allocates pinned memory on the host using RMM

```c
CUVS_EXPORT cuvsError_t cuvsRMMHostAlloc(void** ptr, size_t bytes);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `ptr` | out | `void**` | Pointer to allocated host memory |
| `bytes` | in | `size_t` | Size in bytes to allocate |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrmmhostfree"></a>
### cuvsRMMHostFree

Deallocates pinned memory on the host using RMM

```c
CUVS_EXPORT cuvsError_t cuvsRMMHostFree(void* ptr, size_t bytes);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `ptr` | in | `void*` | Pointer to allocated host memory to free |
| `bytes` | in | `size_t` | Size in bytes to deallocate |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsversionget"></a>
### cuvsVersionGet

Get the version of the cuVS library

```c
CUVS_EXPORT cuvsError_t cuvsVersionGet(uint16_t* major, uint16_t* minor, uint16_t* patch);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `major` | out | `uint16_t*` | Major version |
| `minor` | out | `uint16_t*` | Minor version |
| `patch` | out | `uint16_t*` | Patch version |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmatrixcopy"></a>
### cuvsMatrixCopy

Copy a matrix

```c
CUVS_EXPORT cuvsError_t cuvsMatrixCopy(cuvsResources_t res, DLManagedTensor* src, DLManagedTensor* dst);
```

This function copies a matrix from dst to src. This lets you copy a matrix from device memory to host memory (or vice versa), while accounting for differences in strides.

Both src and dst must have the same shape and dtype, but can have different strides and device type. The memory for the output dst tensor must already be allocated and the tensor initialized.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `src` | in | `DLManagedTensor*` | Pointer to DLManagedTensor to copy |
| `dst` | out | `DLManagedTensor*` | Pointer to DLManagedTensor to receive copy of data |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmatrixslicerows"></a>
### cuvsMatrixSliceRows

Slices rows from a matrix

```c
CUVS_EXPORT cuvsError_t cuvsMatrixSliceRows(
cuvsResources_t res, DLManagedTensor* src, int64_t start, int64_t end, DLManagedTensor* dst);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `src` | in | `DLManagedTensor*` | Pointer to DLManagedTensor to copy |
| `start` | in | `int64_t` | First row index to include in the output |
| `end` | in | `int64_t` | Last row index to include in the output |
| `dst` | out | `DLManagedTensor*` | Pointer to DLManagedTensor to receive slice from matrix |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
