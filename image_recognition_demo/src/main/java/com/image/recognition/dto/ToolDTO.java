package com.image.recognition.dto;

import org.springframework.hateoas.RepresentationModel;

import com.image.net.model_train_tool.ml.ToolType;

public class ToolDTO extends RepresentationModel<ToolDTO> {
	
	ToolType type;
	double probability;
	
	
	public ToolDTO(ToolType type, double probability) {
		this.type = type;
		this.probability = probability;
	}
	
	public ToolType getType() {
		return type;
	}
	
	public void setType(ToolType type) {
		this.type = type;
	}
	
	public double getProbability() {
		return probability;
	}
	
	public void setProbability(double probability) {
		this.probability = probability;
	}
}
