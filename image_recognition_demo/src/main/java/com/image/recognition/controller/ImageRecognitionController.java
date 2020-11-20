package com.image.recognition.controller;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.linkTo;
import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.methodOn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.BaseImageLoader;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.CollectionModel;
import org.springframework.hateoas.Link;
import org.springframework.hateoas.server.mvc.WebMvcLinkBuilder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.image.net.model_train_tool.ml.AnimalType;
import com.image.recognition.dto.AnimalDTO;
import com.image.recognition.exception.InvalidFileFormatException;
import com.image.recognition.service.ImageRecognitionService;

import javafx.util.Pair;
	

@RestController
public class ImageRecognitionController {

	private final static Logger LOGGER = Logger.getLogger(ImageRecognitionController.class.getName());
	private String[] allowedFileTypes = BaseImageLoader.ALLOWED_FORMATS;
	
	
	@Autowired
	private ImageRecognitionService imageRecognitionService;
	
    @GetMapping("/detect")
    public AnimalDTO getAnimalClassification(@RequestParam("imageFile") MultipartFile imageFile) throws IOException {
    	LOGGER.info("Type of image file: " + imageFile.getContentType());
    	
    	if (isFileFormatAllowed(imageFile)) {
    		File file = convertMultipartFileToFile(imageFile);
	    	Pair<AnimalType, Double> typeOfAnimal = imageRecognitionService.detectAnimal(file, 0.5);
	    	
	    	LOGGER.info(typeOfAnimal.getKey().getType());
	    	LOGGER.info(typeOfAnimal.getValue().toString());
	    	
	    	AnimalDTO animalDto = new AnimalDTO(typeOfAnimal.getKey(), typeOfAnimal.getValue());
	    	
	    	/* Adding links to related endpoints and self */
	    	WebMvcLinkBuilder linkToClassificationLabels = linkTo(methodOn(this.getClass()).getAnimalLabels());
	    	Link selfRelLink = linkTo(ImageRecognitionController.class).withSelfRel();
	    	animalDto.add(selfRelLink);
	    	animalDto.add(linkToClassificationLabels.withRel("all-classification-labels"));
	    	
	    	return animalDto;
    	} else {
    		throw new InvalidFileFormatException("Invalid file format. Allowed formats: tif, jpg, png, jpeg, bmp, JPEG, JPG, TIF, PNG");
    	}
    }
    
    @GetMapping("/labels")
    public CollectionModel<AnimalType> getAnimalLabels() {
    	List<AnimalType> animalLabels = Arrays.asList(AnimalType.values());
    	Link selfRelLink = linkTo(ImageRecognitionController.class).withSelfRel();
    	CollectionModel<AnimalType> labels = CollectionModel.of(animalLabels, selfRelLink);
    	return labels;
    }
    
    private File convertMultipartFileToFile(MultipartFile mpFile) {
    	File file = new File("src/main/resources/temp/" + mpFile.getOriginalFilename());
   	 	LOGGER.info(file.getName());
   	 	LOGGER.info(file.getPath());
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
    
    private boolean isFileFormatAllowed(MultipartFile mpFile) {
    	String fileType = FilenameUtils.getExtension(mpFile.getOriginalFilename()).toLowerCase();
    	for(int i = 0; i < allowedFileTypes.length; i++) {
    		if (fileType.equals(allowedFileTypes[i].toLowerCase())) {
    			return true;
    		}
    	}
    	return false;
    }
}
