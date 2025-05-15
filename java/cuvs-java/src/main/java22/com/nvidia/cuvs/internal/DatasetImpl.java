package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.Dataset;

public class DatasetImpl implements Dataset {
	private final Arena arena;
	protected final MemorySegment seg;
	private final int size;
	private final int dimensions;
	private int current = 0;
	
	public DatasetImpl(int size, int dimensions) {
		this.size = size;
		this.dimensions = dimensions;
		
	    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(size * dimensions, C_FLOAT);
	    //seg = this.cuVSResources.getArena().allocate(dataMemoryLayout);
	    
	    this.arena = Arena.ofShared();
	    seg = arena.allocate(dataMemoryLayout);
	}

	@Override
	public void addVector(float[] vector) {
	   if (current >= size) throw new ArrayIndexOutOfBoundsException();
       MemorySegment.copy(vector, 0, seg, C_FLOAT, ((current++) * dimensions * C_FLOAT.byteSize()), (int) dimensions);
	}

	@Override
	public void close() {
		arena.close();
	}

	@Override
	public int getSize() {
		return size;
	}

	@Override
	public int getDimensions() {
		return dimensions;
	}

}
