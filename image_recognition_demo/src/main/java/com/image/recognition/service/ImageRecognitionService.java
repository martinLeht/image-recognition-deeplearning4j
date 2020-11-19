package com.image.recognition.service;

import java.io.File;
import java.io.IOException;

import com.image.net.model_train_tool.ml.AnimalType;

import javafx.util.Pair;


public interface ImageRecognitionService {
	public Pair<AnimalType, Double> detectAnimal(File file, Double threshold) throws IOException;
}
