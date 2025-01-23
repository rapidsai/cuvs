package com.nvidia.cuvs;

public class GPUException extends RuntimeException {

	private static final long serialVersionUID = 4406787247551929480L;
	
	public GPUException(Exception ex) {
		super (ex);
	}

	public GPUException(String message) {
		super (message);
	}
}
