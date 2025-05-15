package com.nvidia.cuvs;

import java.util.Objects;

import com.nvidia.cuvs.spi.CuVSProvider;

public interface Dataset extends AutoCloseable {

	public void addVector(float[] vector);
	
	static Dataset create(int size, int dimensions) {
		return CuVSProvider.provider().newDataset(size, dimensions);
	}

	public int getSize();
	
	public int getDimensions();
}
