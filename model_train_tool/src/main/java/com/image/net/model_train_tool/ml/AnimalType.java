package com.image.net.model_train_tool.ml;

public enum AnimalType {
	CAT("Cat"),
    DOG("Dog"),
    UNKNOWN("Unknown");
	
	private String type;
	
	AnimalType(String type) {
		this.type = type;
	}
	
	public String getType() {
		return type;
	}
}
