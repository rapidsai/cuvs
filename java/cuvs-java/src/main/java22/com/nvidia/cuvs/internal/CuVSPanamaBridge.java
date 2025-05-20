package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkError;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.internal.common.LinkerHelper;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.DLDevice;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import com.nvidia.cuvs.internal.panama.PanamaFFMAPI;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndex;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqParams;

public class CuVSPanamaBridge {

  private static final long C_INT_BYTE_SIZE = LinkerHelper.C_INT.byteSize();
  private static final long C_FLOAT_BYTE_SIZE = LinkerHelper.C_FLOAT.byteSize();
  private static final long C_LONG_BYTE_SIZE = LinkerHelper.C_LONG.byteSize();

  public static MemorySegment createResources(Arena sharedArena) {
    MemorySegment resourcesMemorySegment = sharedArena.allocate(PanamaFFMAPI.cuvsResources_t);
    var returnValue = PanamaFFMAPI.cuvsResourcesCreate(resourcesMemorySegment);
    checkError(returnValue, "cuvsResourcesCreate");
    return resourcesMemorySegment;
  }

  public static void destroyResources(MemorySegment resourcesMemorySegment) {
    var returnValue = PanamaFFMAPI.cuvsResourcesDestroy(resourcesMemorySegment.get(PanamaFFMAPI.cuvsResources_t, 0));
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

    long datasetShape[] = { rows, dimensions };
    MemorySegment datasetTensor = prepareTensor(arena, dataset, datasetShape, 2, 32, 2, 2);

    MemorySegment index = arena.allocate(PanamaFFMAPI.cuvsCagraIndex_t);
    returnValue = PanamaFFMAPI.cuvsCagraIndexCreate(index);

    if (cuvsCagraIndexParams.build_algo(index_params) == 1) { // when build algo is IVF_PQ
      MemorySegment cuvsIvfPqIndexParamsMS = cuvsIvfPqParams
          .ivf_pq_build_params(cuvsCagraIndexParams.graph_build_params(index_params));
      int n_lists = cuvsIvfPqIndexParams.n_lists(cuvsIvfPqIndexParamsMS);
      // As rows cannot be less than n_lists value so trim down.
      cuvsIvfPqIndexParams.n_lists(cuvsIvfPqIndexParamsMS, (int) (rows < n_lists ? rows : n_lists));
    }

    cuvsCagraIndexParams.compression(index_params, compression_params);
    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);

    returnValue = PanamaFFMAPI.cuvsCagraBuild(cuvs_res, index_params, datasetTensor, index);
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
    var returnValue = PanamaFFMAPI.cuvsCagraSerialize(cuvs_res, arena.allocateFrom(filename), index, true);
    checkError(returnValue, "cuvsCagraSerialize");
  }

  public static void deserializeCagraIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment index,
      String filename) {
    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    var returnValue = PanamaFFMAPI.cuvsCagraDeserialize(cuvs_res, arena.allocateFrom(filename), index);
    checkError(returnValue, "cuvsCagraDeserialize");
  }

  public static void searchCagraIndex(Arena arena, MemorySegment index, MemorySegment queries, int topk, long n_queries,
      int dimensions, MemorySegment cuvs_resources, MemorySegment neighbors_h, MemorySegment distances_h,
      MemorySegment search_params) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment stream = arena.allocate(PanamaFFMAPI.cudaStream_t);
    int returnValue = PanamaFFMAPI.cuvsStreamGet(cuvs_res, stream);

    MemorySegment queries_d = arena.allocate(LinkerHelper.C_POINTER);
    MemorySegment neighbors = arena.allocate(LinkerHelper.C_POINTER);
    MemorySegment distances = arena.allocate(LinkerHelper.C_POINTER);

    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, queries_d, C_FLOAT_BYTE_SIZE * n_queries * dimensions);
    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, neighbors, C_INT_BYTE_SIZE * n_queries * topk);
    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, distances, C_FLOAT_BYTE_SIZE * n_queries * topk);

    MemorySegment queries_d_p = queries_d.get(LinkerHelper.C_POINTER, 0);
    MemorySegment neighbors_p = neighbors.get(LinkerHelper.C_POINTER, 0);
    MemorySegment distances_p = distances.get(LinkerHelper.C_POINTER, 0);

    returnValue = PanamaFFMAPI.cudaMemcpy(queries_d_p, queries, C_FLOAT_BYTE_SIZE * n_queries * dimensions, 4);

    long queries_shape[] = { n_queries, dimensions };
    MemorySegment queries_tensor = prepareTensor(arena, queries_d_p, queries_shape, 2, 32, 2, 2);

    long neighbors_shape[] = { n_queries, topk };
    MemorySegment neighbors_tensor = prepareTensor(arena, neighbors_p, neighbors_shape, 1, 32, 2, 2);

    long distances_shape[] = { n_queries, topk };
    MemorySegment distances_tensor = prepareTensor(arena, distances_p, distances_shape, 2, 32, 2, 2);

    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);

    MemorySegment filter = cuvsFilter.allocate(arena);
    cuvsFilter.type(filter, 0);
    cuvsFilter.addr(filter, 0);

    returnValue = PanamaFFMAPI.cuvsCagraSearch(cuvs_res, search_params, index, queries_tensor, neighbors_tensor,
        distances_tensor, filter);

    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);

    returnValue = PanamaFFMAPI.cudaMemcpy(neighbors_h, neighbors_p, C_INT_BYTE_SIZE * n_queries * topk, 4);
    returnValue = PanamaFFMAPI.cudaMemcpy(distances_h, distances_p, C_FLOAT_BYTE_SIZE * n_queries * topk, 4);

    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, distances_p, C_FLOAT_BYTE_SIZE * n_queries * topk);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, neighbors_p, C_INT_BYTE_SIZE * n_queries * topk);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, queries_d_p, C_FLOAT_BYTE_SIZE * n_queries * dimensions);
  }

  public static MemorySegment buildBruteForceIndex(Arena arena, MemorySegment dataset, long rows, long dimensions,
      MemorySegment cuvs_resources, int n_writer_threads) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment stream = arena.allocate(PanamaFFMAPI.cudaStream_t);
    var returnValue = PanamaFFMAPI.cuvsStreamGet(cuvs_res, stream);

    MemorySegment dataset_d = arena.allocate(LinkerHelper.C_POINTER);

    long dataset_bytes = C_FLOAT_BYTE_SIZE * rows * dimensions;
    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, dataset_d, dataset_bytes);

    MemorySegment dataset_d_p = dataset_d.get(LinkerHelper.C_POINTER, 0);

    returnValue = PanamaFFMAPI.cudaMemcpy(dataset_d_p, dataset, dataset_bytes, 4);

    long dataset_shape[] = { rows, dimensions };
    MemorySegment dataset_tensor = prepareTensor(arena, dataset_d_p, dataset_shape, 2, 32, 2, 2);

    MemorySegment index = arena.allocate(PanamaFFMAPI.cuvsBruteForceIndex_t);

    returnValue = PanamaFFMAPI.cuvsBruteForceIndexCreate(index);

    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);

    returnValue = PanamaFFMAPI.cuvsBruteForceBuild(cuvs_res, dataset_tensor, 0, 0.0f, index);

    // TODO: omp_set_num_threads is pending for now.
    return index;
  }

  public static void destroyBruteForceIndex(MemorySegment index) {
    int returnValue = PanamaFFMAPI.cuvsBruteForceIndexDestroy(index);
    checkError(returnValue, "cuvsBruteForceIndexDestroy");
  }

  public static void searchBruteForceIndex(Arena arena, MemorySegment index, MemorySegment queries, int topk,
      long n_queries, int dimensions, MemorySegment cuvs_resources, MemorySegment neighbors_h,
      MemorySegment distances_h, MemorySegment prefilter_data, long prefilter_data_length) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment stream = arena.allocate(PanamaFFMAPI.cudaStream_t);
    var returnValue = PanamaFFMAPI.cuvsStreamGet(cuvs_res, stream);

    MemorySegment queries_d = arena.allocate(LinkerHelper.C_POINTER);
    MemorySegment neighbors = arena.allocate(LinkerHelper.C_POINTER);
    MemorySegment distances = arena.allocate(LinkerHelper.C_POINTER);

    MemorySegment prefilter_d = arena.allocate(LinkerHelper.C_POINTER);
    MemorySegment prefilter_d_p = MemorySegment.NULL;
    long prefilter_len = 0;

    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, queries_d, C_FLOAT_BYTE_SIZE * n_queries * dimensions);
    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, neighbors, C_LONG_BYTE_SIZE * n_queries * topk);
    returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, distances, C_FLOAT_BYTE_SIZE * n_queries * topk);

    MemorySegment queries_d_p = queries_d.get(LinkerHelper.C_POINTER, 0);
    MemorySegment neighbors_p = neighbors.get(LinkerHelper.C_POINTER, 0);
    MemorySegment distances_p = distances.get(LinkerHelper.C_POINTER, 0);

    returnValue = PanamaFFMAPI.cudaMemcpy(queries_d_p, queries, C_FLOAT_BYTE_SIZE * n_queries * dimensions, 4);

    long queries_shape[] = { n_queries, dimensions };
    MemorySegment queries_tensor = prepareTensor(arena, queries_d_p, queries_shape, 2, 32, 2, 2);

    long neighbors_shape[] = { n_queries, topk };
    MemorySegment neighbors_tensor = prepareTensor(arena, neighbors_p, neighbors_shape, 0, 64, 2, 2);

    long distances_shape[] = { n_queries, topk };
    MemorySegment distances_tensor = prepareTensor(arena, distances_p, distances_shape, 2, 32, 2, 2);

    MemorySegment prefilter = cuvsFilter.allocate(arena);

    MemorySegment prefilter_tensor;

    if (prefilter_data == MemorySegment.NULL) {
      cuvsFilter.type(prefilter, 0); // NO_FILTER
      cuvsFilter.addr(prefilter, 0);
    } else {
      long prefilter_shape[] = { (prefilter_data_length + 31) / 32 };
      prefilter_len = prefilter_shape[0];

      returnValue = PanamaFFMAPI.cuvsRMMAlloc(cuvs_res, prefilter_d, C_INT_BYTE_SIZE * prefilter_len);

      prefilter_d_p = prefilter_d.get(LinkerHelper.C_POINTER, 0);

      returnValue = PanamaFFMAPI.cudaMemcpy(prefilter_d_p, prefilter_data, C_INT_BYTE_SIZE * prefilter_len, 1);

      prefilter_tensor = prepareTensor(arena, prefilter_d_p, prefilter_shape, 1, 32, 1, 2);

      cuvsFilter.type(prefilter, 2);
      cuvsFilter.addr(prefilter, prefilter_tensor.address());
    }

    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);

    returnValue = PanamaFFMAPI.cuvsBruteForceSearch(cuvs_res, index, queries_tensor, neighbors_tensor, distances_tensor,
        prefilter);

    returnValue = PanamaFFMAPI.cuvsStreamSync(cuvs_res);

    returnValue = PanamaFFMAPI.cudaMemcpy(neighbors_h, neighbors_p, C_LONG_BYTE_SIZE * n_queries * topk, 4);
    returnValue = PanamaFFMAPI.cudaMemcpy(distances_h, distances_p, C_FLOAT_BYTE_SIZE * n_queries * topk, 4);

    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, neighbors_p, C_LONG_BYTE_SIZE * n_queries * topk);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, distances_p, C_FLOAT_BYTE_SIZE * n_queries * topk);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, queries_d_p, C_FLOAT_BYTE_SIZE * n_queries * dimensions);
    returnValue = PanamaFFMAPI.cuvsRMMFree(cuvs_res, prefilter_d_p, C_INT_BYTE_SIZE * prefilter_len);
  }

  public static void serializeBruteForceIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment index,
      String filename) {
    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    int returnValue = PanamaFFMAPI.cuvsBruteForceSerialize(cuvs_res, arena.allocateFrom(filename), index);
    checkError(returnValue, "cuvsBruteForceSerialize");
  }

  public static void deserializeBruteForceIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment index,
      String filename) {
    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    int returnValue = PanamaFFMAPI.cuvsBruteForceDeserialize(cuvs_res, arena.allocateFrom(filename), index);
    checkError(returnValue, "cuvsBruteForceDeserialize");
  }

  public static void serializeCagraIndexToHnsw(MemorySegment cuvs_resources, MemorySegment file_path, MemorySegment index) {
    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    int returnValue = PanamaFFMAPI.cuvsCagraSerializeToHnswlib(cuvs_res, file_path, index);
    checkError(returnValue, "cuvsCagraSerializeToHnswlib");
  }

  public static MemorySegment deserializeHnswIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment file_path,
      MemorySegment hnsw_params, int vector_dimension) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);
    MemorySegment hnsw_index = cuvsHnswIndex.allocate(arena);
    int returnValue = PanamaFFMAPI.cuvsHnswIndexCreate(hnsw_index);

    MemorySegment dtype = DLDataType.allocate(arena);
    DLDataType.bits(dtype, (byte) 32);
    DLDataType.code(dtype, (byte) 2); // kDLFloat
    DLDataType.lanes(dtype, (byte) 1);

    cuvsHnswIndex.dtype(hnsw_index, dtype);

    returnValue = PanamaFFMAPI.cuvsHnswDeserialize(cuvs_res, hnsw_params, file_path, vector_dimension, 0, hnsw_index);

    return hnsw_index;
  }

  public static void searchHnswIndex(Arena arena, MemorySegment cuvs_resources, MemorySegment hnsw_index,
      MemorySegment search_params, MemorySegment neighbors_h, MemorySegment distances_h, MemorySegment queries,
      int topk, int query_dimension, int n_queries) {

    long cuvs_res = cuvs_resources.get(PanamaFFMAPI.cuvsResources_t, 0);

    long queries_shape[] = { n_queries, query_dimension };
    MemorySegment queries_tensor = prepareTensor(arena, queries, queries_shape, 2, 32, 2, 1);

    long neighbors_shape[] = { n_queries, topk };
    MemorySegment neighbors_tensor = prepareTensor(arena, neighbors_h, neighbors_shape, 1, 64, 2, 1);

    long distances_shape[] = { n_queries, topk };
    MemorySegment distances_tensor = prepareTensor(arena, distances_h, distances_shape, 2, 32, 2, 1);

    int returnValue = PanamaFFMAPI.cuvsHnswSearch(cuvs_res, search_params, hnsw_index, queries_tensor, neighbors_tensor,
        distances_tensor);
  }

  public static void destroyHnswIndex(MemorySegment hnsw_index) {
    int returnValue = PanamaFFMAPI.cuvsHnswIndexDestroy(hnsw_index);
    checkError(returnValue, "cuvsHnswIndexDestroy");
  }

}
