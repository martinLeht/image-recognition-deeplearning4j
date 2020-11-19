package com.image.recognition.controller;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.logging.Logger;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.EntityModel;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.image.net.model_train_tool.ml.AnimalType;
import com.image.recognition.dto.AnimalDTO;
import com.image.recognition.service.ImageRecognitionService;

import javafx.util.Pair;
	

@RestController
public class ImageRecognitionController {

	private final static Logger LOGGER = Logger.getLogger(ImageRecognitionController.class.getName());
	
	@Autowired
	private ImageRecognitionService imageRecognitionService;
	
    @GetMapping("/recognition")
    public EntityModel<AnimalDTO> getAnimalClassification(@RequestParam("imageFile") MultipartFile imageFile) throws IOException {
    	
    	File file = convertMultipartFileToFile(imageFile);
    	Pair<AnimalType, Double> typeOfAnimal = imageRecognitionService.detectAnimal(file, 0.5);
    	LOGGER.info(typeOfAnimal.getKey().getType());
    	LOGGER.info(typeOfAnimal.getValue().toString());
    	
    	AnimalDTO animalDto = new AnimalDTO(typeOfAnimal.getKey(), typeOfAnimal.getValue());
    	
    	EntityModel<AnimalDTO> response = EntityModel.of(animalDto);
    	return response;
    }
    
    private File convertMultipartFileToFile(MultipartFile mpFile) {
    	File file = new File("src/main/resources/saved/image.jpg");
   	 
    	try {
    		OutputStream os = new FileOutputStream(file);
    	    os.write(mpFile.getBytes());
    	    return file;
    	} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
    }
}
