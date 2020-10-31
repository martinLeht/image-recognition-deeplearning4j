package com.image.recognition.controller;

import java.io.IOException;
import java.util.logging.Logger;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.EntityModel;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.image.recognition.service.ImageRecognitionService;
	

@RestController
public class ImageRecognitionController {

	private final static Logger LOGGER = Logger.getLogger(ImageRecognitionController.class.getName());
	
	@Autowired
	private ImageRecognitionService imageRecognitionService;
	
    @GetMapping("/recognition")
    public EntityModel<String> getRecognition(@RequestParam("imageFile") MultipartFile imageFile) throws IOException {
    	EntityModel<String> response = EntityModel.of("Successfully processed image recognition");
    	return response;
    }
}
