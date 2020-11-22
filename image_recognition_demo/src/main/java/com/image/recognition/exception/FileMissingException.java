package com.image.recognition.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.BAD_REQUEST)
public class FileMissingException extends RuntimeException {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8001415277311091129L;

	public FileMissingException(String message) {
		super(message);
	}
}
