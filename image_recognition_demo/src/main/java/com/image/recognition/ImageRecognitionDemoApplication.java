package com.image.recognition;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.zip.Adler32;

import org.apache.commons.io.FileUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;


@SpringBootApplication
public class ImageRecognitionDemoApplication {
	
	private final static Logger LOGGER = Logger.getLogger(ImageRecognitionDemoApplication.class.getName());
	
	/* Path to model that was trained using the cmd tool (ModelTrainerTool) */
	public static String MODEL_PATH = System.getProperty("user.home") + "\\Pictures\\image_rec\\cat_or_dog\\saved\\modelIteration_3900_epoch_2.zip";
	
	
	public static void main(String[] args) throws IOException, InterruptedException {
		File model = new File("src/main/resources/saved/model.zip");
		if (!model.exists() || FileUtils.checksum(model, new Adler32()).getValue() != 3082129141l) {
            model.delete();
            
            LOGGER.info("Fetching and copying model for the first time!");
            LOGGER.info("Original model path: " + MODEL_PATH);
            File modelOrigin = new File(MODEL_PATH);
            
            try {
            	FileUtils.copyFile(modelOrigin, model);
            	LOGGER.info("Copied Model file: " + model);
            	LOGGER.info("Copied Model Absolute path: " + model.getAbsolutePath());
            	LOGGER.info("Copied Model Name: " + model.getName());
            } catch (IOException e) {
            	LOGGER.info("Failed to fetch and copy model");
                throw new RuntimeException(e);
            } finally {
            	LOGGER.info("Model fetched and copied DONE");
            }
        }
		LOGGER.info("Executing SpringBoot Application");
		SpringApplication.run(ImageRecognitionDemoApplication.class, args);
	}

}
