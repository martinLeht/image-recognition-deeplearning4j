package com.image.recognition.service;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.Map.Entry;

import com.image.net.model_train_tool.ml.ToolType;

public interface ToolRecognitionService {
	
	public Map<ToolType, Double> detectTool(File file) throws IOException;
	
	public Optional<Entry<ToolType, Double>> getToolWithHighestProbabilityFromPredictions(Map<ToolType, Double> predictions, Double threshold);
}
