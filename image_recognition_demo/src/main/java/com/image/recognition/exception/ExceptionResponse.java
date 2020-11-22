package com.image.recognition.exception;

import java.util.Date;

public class ExceptionResponse {
	
	private Date timestamp;
	private Integer status;
	private String message;
	private String details;
	
	public ExceptionResponse(Date timestamp, Integer status, String message, String details) {
		super();
		this.timestamp = timestamp;
		this.status = status;
		this.message = message;
		this.details = details;
	}
	
	public Date getTimestamp() {
		return timestamp;
	}
	
	public Integer getStatus() {
		return status;
	}
	
	public String getMessage() {
		return message;
	}
	
	public String getDetails() {
		return details;
	}
	

}
