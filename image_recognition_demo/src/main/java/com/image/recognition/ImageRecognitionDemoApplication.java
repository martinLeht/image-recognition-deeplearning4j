package com.image.recognition;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Logger;
import java.util.zip.Adler32;

import org.apache.commons.io.FileUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.image.recognition.ml.RunnableImageNetVGG16;

@SpringBootApplication
public class ImageRecognitionDemoApplication {
	
	private final static Logger LOGGER = Logger.getLogger(ImageRecognitionDemoApplication.class.getName());
	
	/* Path to model that was trained earlier */
	public static String MODEL_PATH = RunnableImageNetVGG16.DATA_PATH + "/saved2/modelIteration_3900_epoch_2.zip";
	
	
	public static void main(String[] args) throws IOException, InterruptedException {
		File model = new File("src/main/resources/saved/model.zip");
		if (!model.exists() || FileUtils.checksum(model, new Adler32()).getValue() != 3082129141l) {
            model.delete();
            
            LOGGER.info("Training the model for the first time!");
            RunnableImageNetVGG16 runnableImageNet = new RunnableImageNetVGG16();
    		Future future = Executors.newCachedThreadPool().submit(runnableImageNet);
    		while(!future.isDone()) {
    		    Thread.sleep(300);
    		}
            
            LOGGER.info("Fetching and copying model for the first time!");
            File modelOrigin = new File(MODEL_PATH);
            
            try {
            	FileUtils.copyFile(modelOrigin, model);
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
