package com.nvidia.cuvs;

public class LibraryNotFoundException extends RuntimeException {

	private static final long serialVersionUID = 4406787247551929480L;
	
	public LibraryNotFoundException(Exception ex) {
		super (ex);
	}

	public LibraryNotFoundException(String message) {
		super (message);
	}
}
