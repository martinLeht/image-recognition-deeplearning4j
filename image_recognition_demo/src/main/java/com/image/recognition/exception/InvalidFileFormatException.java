package com.image.recognition.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.CONFLICT)
public class InvalidFileFormatException extends RuntimeException {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3418396761816883669L;

	public InvalidFileFormatException(String message) {
		super(message);
	}
}
