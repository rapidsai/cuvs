package com.nvidia.cuvs;

public class LibraryException extends RuntimeException {

  private static final long serialVersionUID = -2311171907713571455L;

  public LibraryException(Exception ex) {
    super(ex);
  }

  public LibraryException(String message) {
    super(message);
  }

  public LibraryException(String message, Exception e) {
    super(message, e);
  }
}
