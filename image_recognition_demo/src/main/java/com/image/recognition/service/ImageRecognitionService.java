package com.image.recognition.service;

import java.io.File;
import java.io.IOException;

import com.image.recognition.ml.AnimalType;

public interface ImageRecognitionService {
	public AnimalType detectAnimal(File file, Double threshold) throws IOException;
}
