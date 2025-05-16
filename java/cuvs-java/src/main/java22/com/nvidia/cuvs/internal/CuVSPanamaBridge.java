package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkError;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.DLDevice;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import com.nvidia.cuvs.internal.panama.PanamaFFMAPI;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqParams;

public class CuVSPanamaBridge {

  public static MemorySegment createResources(Arena sharedArena) {
    MemorySegment resourcesMemorySegment = sharedArena.allocate(PanamaFFMAPI.cuvsResources_t);
    var returnValue = PanamaFFMAPI.cuvsResourcesCreate(resourcesMemorySegment);
    System.out.println("CuVS resources create return value: " + returnValue);
    checkError(returnValue, "cuvsResourcesCreate");
    return resourcesMemorySegment;
  }

  public static void destroyResources(MemorySegment resourcesMemorySegment) {
    var returnValue = PanamaFFMAPI.cuvsResourcesDestroy(resourcesMemorySegment.get(PanamaFFMAPI.cuvsResources_t, 0));
    System.out.println("CuVS resources destroy return value: " + returnValue);
    checkError(returnValue, "cuvsResourcesDestroy");
  }

  public static MemorySegment prepareTensor(Arena arena, MemorySegment data, long[] shape, int code, int bits, int ndim,
      int deviceType) {

    MemorySegment tensor = DLManagedTensor.allocate(arena);
    MemorySegment dlTensor = DLTensor.allocate(arena);

    DLTensor.data(dlTensor, data);

    MemorySegment dlDevice = DLDevice.allocate(arena);
    DLDevice.device_type(dlDevice, deviceType);
    DLTensor.device(dlTensor, dlDevice);

    DLTensor.ndim(dlTensor, ndim);

    MemorySegment dtype = DLDataType.allocate(arena);
    DLDataType.code(dtype, (byte) code);
    DLDataType.bits(dtype, (byte) bits);
    DLDataType.lanes(dtype, (short) 1);
    DLTensor.dtype(dlTensor, dtype);

    DLTensor.shape(dlTensor, Util.buildMemorySegment(arena, shape));

    DLTensor.strides(dlTensor, MemorySegment.NULL);

    DLManagedTensor.dl_tensor(tensor, dlTensor);

    return tensor;
  }

  public static MemorySegment buildCagraIndex(Arena arena, MemorySegment dataset, long rows, long dimensions,
      MemorySegment cuvs_resources, MemorySegment index_params, MemorySegment compression_params,
      int n_writer_threads) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment stream = arena.allocate(PanamaFFMAPI.cudaStream_t);
    var returnValue = PanamaFFMAPI.cuvsStreamGet(cuvs_res, stream);
    System.out.println("cuvsStreamGet return value: " + returnValue);

    long datasetShape[] = { rows, dimensions };
    MemorySegment datasetTensor = prepareTensor(arena, dataset, datasetShape, 2, 32, 2, 2);

    MemorySegment index = arena.allocate(PanamaFFMAPI.cuvsCagraIndex_t);
    returnValue = PanamaFFMAPI.cuvsCagraIndexCreate(index);
    System.out.println("cuvsCagraIndexCreate return value: " + returnValue);

    if (cuvsCagraIndexParams.build_algo(index_params) == 1) { // when build algo is IVF_PQ
      MemorySegment cuvsIvfPqIndexParamsMS = cuvsIvfPqParams
          .ivf_pq_build_params(cuvsCagraIndexParams.graph_build_params(index_params));
      int n_lists = cuvsIvfPqIndexParams.n_lists(cuvsIvfPqIndexParamsMS);
      // As rows cannot be less than n_lists value so trim down.
      cuvsIvfPqIndexParams.n_lists(cuvsIvfPqIndexParamsMS, (int) (rows < n_lists ? rows : n_lists));
    }

    cuvsCagraIndexParams.compression(index_params, compression_params);
    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);
    System.out.println("cuvsStreamSync return value: " + returnValue);

    returnValue = PanamaFFMAPI.cuvsCagraBuild(cuvs_res, index_params, datasetTensor, index);
    System.out.println("cuvsCagraBuild return value: " + returnValue);
    checkError(returnValue, "cuvsCagraBuild");

    // TODO: omp_set_num_threads is pending for now.
    return index;
  }

  public static void destroyCagraIndex(MemorySegment index) {
    int returnValue = PanamaFFMAPI.cuvsCagraIndexDestroy(index);
    checkError(returnValue, "cuvsCagraIndexDestroy");
  }

  public static void serializeCagraIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment index,
      String filename) {
    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment fn = arena.allocateFrom(filename);
    var returnValue = PanamaFFMAPI.cuvsCagraSerialize(cuvs_res, fn, index, true);
    System.out.println("cuvsCagraSerialize return value: " + returnValue);
    checkError(returnValue, "cuvsCagraSerialize");
  }

  public static void deserializeCagraIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment index,
      String filename) {
    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    var returnValue = PanamaFFMAPI.cuvsCagraDeserialize(cuvs_res, arena.allocateFrom(filename), index);
    System.out.println("cuvsCagraDeserialize return value: " + returnValue);
    checkError(returnValue, "cuvsCagraDeserialize");
  }

  public static void searchCagraIndex(Arena arena, MemorySegment index, MemorySegment queries, int topk, long n_queries,
      int dimensions, MemorySegment cuvs_resources, MemorySegment neighbors_h, MemorySegment distances_h,
      MemorySegment search_params) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment stream = arena.allocate(PanamaFFMAPI.cudaStream_t);
    int returnValue = PanamaFFMAPI.cuvsStreamGet(cuvs_res, stream);
    System.out.println("cuvsStreamGet return value: " + returnValue);

    MemorySegment queries_d = arena.allocate(PanamaFFMAPI.C_POINTER);
    MemorySegment neighbors = arena.allocate(PanamaFFMAPI.C_POINTER);
    MemorySegment distances = arena.allocate(PanamaFFMAPI.C_POINTER);

    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, queries_d, PanamaFFMAPI.C_FLOAT.byteSize() * n_queries * dimensions);
    System.out.println("cuvsRMMAlloc return value: " + returnValue);

    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, neighbors, PanamaFFMAPI.C_INT.byteSize() * n_queries * topk);
    System.out.println("cuvsRMMAlloc return value: " + returnValue);

    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, distances, PanamaFFMAPI.C_FLOAT.byteSize() * n_queries * topk);
    System.out.println("cuvsRMMAlloc return value: " + returnValue);

    returnValue = PanamaFFMAPI.cudaMemcpy(queries_d, queries, PanamaFFMAPI.C_FLOAT.byteSize() * n_queries * dimensions,
        4);

    System.out.println("cudaMemcpy return value: " + returnValue);

    long queries_shape[] = { n_queries, dimensions };
    MemorySegment queries_tensor = prepareTensor(arena, queries_d, queries_shape, 2, 32, 2, 2);

    long neighbors_shape[] = { n_queries, topk };
    MemorySegment neighbors_tensor = prepareTensor(arena, neighbors, neighbors_shape, 1, 32, 2, 2);

    long distances_shape[] = { n_queries, topk };
    MemorySegment distances_tensor = prepareTensor(arena, distances, distances_shape, 2, 32, 2, 2);

    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);
    System.out.println("cuvsStreamSync return value: " + returnValue);

    MemorySegment filter = cuvsFilter.allocate(arena);
    cuvsFilter.type(filter, 0);
    cuvsFilter.addr(filter, 0);

    returnValue = PanamaFFMAPI.cuvsCagraSearch(cuvs_res, search_params, index, queries_tensor, neighbors_tensor,
        distances_tensor, filter);

    System.out.println("cuvsCagraSearch return value: " + returnValue);

    returnValue = PanamaFFMAPI.cudaMemcpy(neighbors_h, neighbors, PanamaFFMAPI.C_INT.byteSize() * n_queries * topk, 4);
    System.out.println("cudaMemcpy return value: " + returnValue);
    returnValue = PanamaFFMAPI.cudaMemcpy(distances_h, distances, PanamaFFMAPI.C_FLOAT.byteSize() * n_queries * topk, 4);
    System.out.println("cudaMemcpy return value: " + returnValue);

    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, distances, PanamaFFMAPI.C_FLOAT.byteSize() * n_queries * topk);
    System.out.println("cuvsRMMFree return value: " + returnValue);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, neighbors, PanamaFFMAPI.C_INT.byteSize() * n_queries * topk);
    System.out.println("cuvsRMMFree return value: " + returnValue);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, queries_d, PanamaFFMAPI.C_FLOAT.byteSize() * n_queries * dimensions);
    System.out.println("cuvsRMMFree return value: " + returnValue);

  }
}
