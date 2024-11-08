package com.nvidia.cuvs.cagra;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.UUID;

public class CuVSIndex {

  private IndexParams indexParams;
  private final float[][] dataset;
  private final CuVSResources res;
  private CagraIndexReference ref;
  private ANNAlgorithms algo;

  public enum ANNAlgorithms {
    BRUTEFORCE, CAGRA, IVF_PQ, IVF_FLAT
  }

  Linker linker;
  Arena arena;
  MethodHandle cresMH;
  MethodHandle indexMH;
  MethodHandle searchMH;
  MethodHandle serializeMH;
  MethodHandle deserializeMH;
  MemorySegment dataMS;
  SymbolLookup bridge;

  /**
   * 
   * @param indexParams
   * @param dataset
   * @param mapping
   * @param res
   * @throws Throwable
   */
  private CuVSIndex(IndexParams indexParams, float[][] dataset, CuVSResources res, ANNAlgorithms algo)
      throws Throwable {
    this.indexParams = indexParams;
    this.dataset = dataset;
    this.init();
    this.res = res;
    this.ref = build();
    this.algo = algo;
  }

  /**
   * 
   * @param in
   * @param res
   * @throws Throwable
   */
  private CuVSIndex(InputStream in, CuVSResources res) throws Throwable {
    this.indexParams = null;
    this.dataset = null;
    this.res = res;
    this.init();
    this.ref = deserialize(in);
  }

  /**
   * 
   * @throws Throwable
   */
  private void init() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File wd = new File(System.getProperty("user.dir"));
    bridge = SymbolLookup.libraryLookup(wd.getParent() + "/api-sys/build/libcuvs_wrapper.so", arena);

    indexMH = linker.downcallHandle(bridge.findOrThrow("build_index"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    searchMH = linker.downcallHandle(bridge.findOrThrow("search_index"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"),
            linker.canonicalLayouts().get("long"), linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    serializeMH = linker.downcallHandle(bridge.findOrThrow("serialize_index"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    deserializeMH = linker.downcallHandle(bridge.findOrThrow("deserialize_index"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

  }

  /**
   * Java String -> C char*
   * 
   * @param str
   * @return MemorySegment
   */
  public MemorySegment getStringSegment(StringBuilder str) {
    str.append('\0');
    MemoryLayout sq = MemoryLayout.sequenceLayout(str.length(), linker.canonicalLayouts().get("char"));
    MemorySegment fln = arena.allocate(sq);

    for (int i = 0; i < str.length(); i++) {
      VarHandle flnVH = sq.varHandle(PathElement.sequenceElement(i));
      flnVH.set(fln, 0L, (byte) str.charAt(i));
    }
    return fln;
  }

  /**
   * 
   * @param data
   * @return
   */
  private MemorySegment getMemorySegment(float[][] data) {
    long rows = data.length;
    long cols = data[0].length;

    MemoryLayout dataML = MemoryLayout.sequenceLayout(rows,
        MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
    MemorySegment dataMS = arena.allocate(dataML);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        VarHandle element = dataML.arrayElementVarHandle(PathElement.sequenceElement(r),
            PathElement.sequenceElement(c));
        element.set(dataMS, 0, 0, data[r][c]);
      }
    }

    return dataMS;
  }

  /**
   * 
   * @return
   * @throws Throwable
   */
  private CagraIndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = dataset[0].length;
    MemoryLayout rvML = linker.canonicalLayouts().get("int");
    MemorySegment rvMS = arena.allocate(rvML);

    ref = new CagraIndexReference((MemorySegment) indexMH.invokeExact(getMemorySegment(dataset), rows, cols,
        res.resource, rvMS, indexParams.indexParamsMS));

    System.out.println("Build call return value: " + rvMS.get(ValueLayout.JAVA_INT, 0));

    return ref;
  }

  /**
   * 
   * @param params
   * @param queryVectors
   * @return
   * @throws Throwable
   */
  public SearchResult search(CuVSQuery query) throws Throwable {

    SequenceLayout neighborsSL = MemoryLayout.sequenceLayout(50, linker.canonicalLayouts().get("int"));
    SequenceLayout distancesSL = MemoryLayout.sequenceLayout(50, linker.canonicalLayouts().get("float"));
    MemorySegment neighborsMS = arena.allocate(neighborsSL);
    MemorySegment distancesMS = arena.allocate(distancesSL);
    MemoryLayout rvML = linker.canonicalLayouts().get("int");
    MemorySegment rvMS = arena.allocate(rvML);

    searchMH.invokeExact(ref.indexMemorySegment, getMemorySegment(query.queryVectors), 2, 4L, 2L, res.resource,
        neighborsMS, distancesMS, rvMS, query.searchParams.searchParamsMS);

    System.out.println("Search call return value: " + rvMS.get(ValueLayout.JAVA_INT, 0));

    return new SearchResult(neighborsSL, distancesSL, neighborsMS, distancesMS, 2, query.mapping);
  }

  /**
   * 
   * @param out
   * @param tmpFilePath
   * @throws Throwable
   */
  public void serialize(OutputStream out, String tmpFilePath) throws Throwable {
    MemoryLayout rvML = linker.canonicalLayouts().get("int");
    MemorySegment rvMS = arena.allocate(rvML);
    serializeMH.invokeExact(res.resource, ref.indexMemorySegment, rvMS,
        getStringSegment(new StringBuilder(tmpFilePath)));
    System.out.println("Serialize call return value: " + rvMS.get(ValueLayout.JAVA_INT, 0));

    File tempFile = new File(tmpFilePath);
    FileInputStream is = new FileInputStream(tempFile);
    byte[] chunk = new byte[1024];
    int chunkLen = 0;
    while ((chunkLen = is.read(chunk)) != -1) {
      out.write(chunk, 0, chunkLen);
    }
    is.close();
    tempFile.delete();
  }

  /**
   * 
   * @param in
   * @return
   * @throws Throwable
   */
  private CagraIndexReference deserialize(InputStream in) throws Throwable {
    MemoryLayout rvML = linker.canonicalLayouts().get("int");
    MemorySegment rvMS = arena.allocate(rvML);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".cag";
    CagraIndexReference ref = new CagraIndexReference();

    File tempFile = new File(tmpIndexFile);
    FileOutputStream out = new FileOutputStream(tempFile);
    byte[] chunk = new byte[1024];
    int chunkLen = 0;
    while ((chunkLen = in.read(chunk)) != -1) {
      out.write(chunk, 0, chunkLen);
    }
    deserializeMH.invokeExact(res.resource, ref.indexMemorySegment, rvMS,
        getStringSegment(new StringBuilder(tmpIndexFile)));

    in.close();
    out.close();
    tempFile.delete();

    return ref;
  }

  /**
   * 
   * @return
   */
  public IndexParams getParams() {
    return indexParams;
  }

  /**
   * 
   * @return
   */
  public PointerToDataset getDataset() {
    return null;
  }

  /**
   * 
   * @return
   */
  public CuVSResources getResources() {
    return res;
  }

  public static class Builder {
    private IndexParams indexParams;
    float[][] dataset;
    CuVSResources res;
    ANNAlgorithms algo = ANNAlgorithms.CAGRA;

    InputStream in;

    /**
     * 
     * @param res
     */
    public Builder(CuVSResources res) {
      this.res = res;
    }

    /**
     * 
     * @param in
     * @return
     */
    public Builder from(InputStream in) {
      this.in = in;
      return this;
    }

    /**
     * 
     * @param dataset
     * @return
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * 
     * @param params
     * @return
     */
    public Builder withIndexParams(IndexParams indexParams) {
      this.indexParams = indexParams;
      return this;
    }

    /**
     * 
     * @param params
     * @return
     */
    public Builder withANNAlgorithm(ANNAlgorithms algo) {
      this.algo = algo;
      return this;
    }

    /**
     * 
     * @return
     * @throws Throwable
     */
    public CuVSIndex build() throws Throwable {
      if (in != null) {
        return new CuVSIndex(in, res);
      } else {
        return new CuVSIndex(indexParams, dataset, res, algo);
      }
    }
  }

}
