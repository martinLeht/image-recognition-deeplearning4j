package com.image.recognition.service;

import java.io.File;
import java.io.IOException;

import com.image.net.model_train_tool.ml.AnimalType;


public interface ImageRecognitionService {
	public AnimalType detectAnimal(File file, Double threshold) throws IOException;
}
