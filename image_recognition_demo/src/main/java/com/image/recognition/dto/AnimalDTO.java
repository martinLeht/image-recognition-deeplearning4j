package com.image.recognition.dto;

import com.image.net.model_train_tool.ml.AnimalType;

public class AnimalDTO {
	
	AnimalType type;
	double probability;
	
	
	public AnimalDTO(AnimalType type, double probability) {
		this.type = type;
		this.probability = probability;
	}
	public AnimalType getType() {
		return type;
	}
	public void setType(AnimalType type) {
		this.type = type;
	}
	public double getProbability() {
		return probability;
	}
	public void setProbability(double probability) {
		this.probability = probability;
	}
	
	
}
