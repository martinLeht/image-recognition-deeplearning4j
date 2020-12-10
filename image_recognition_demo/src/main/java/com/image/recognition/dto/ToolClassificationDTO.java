package com.image.recognition.dto;

import java.util.Map;

import org.springframework.hateoas.RepresentationModel;

import com.image.net.model_train_tool.ml.ToolType;

public class ToolClassificationDTO extends RepresentationModel<ToolClassificationDTO> {
	
	ToolDTO classifiedAs;
	Map<ToolType, Double> allPredictions;
	
	public ToolClassificationDTO(ToolDTO classifiedAs, Map<ToolType, Double> allPredictions) {
		this.classifiedAs = classifiedAs;
		this.allPredictions = allPredictions;
	}
	
	public ToolDTO getClassifiedAs() {
		return classifiedAs;
	}
	
	public void setClassifiedAs(ToolDTO  classifiedAs) {
		this.classifiedAs = classifiedAs;
	}
	
	public Map<ToolType, Double> getPredictions() {
		return allPredictions;
	}
	
	public void setProbability(Map<ToolType, Double> allPredictions) {
		this.allPredictions = allPredictions;
	}
}
