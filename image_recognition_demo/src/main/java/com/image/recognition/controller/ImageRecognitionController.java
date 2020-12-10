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
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.logging.Logger;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.BaseImageLoader;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.CollectionModel;
import org.springframework.hateoas.Link;
import org.springframework.hateoas.server.mvc.WebMvcLinkBuilder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.image.net.model_train_tool.ml.AnimalType;
import com.image.net.model_train_tool.ml.ToolType;
import com.image.recognition.dto.AnimalDTO;
import com.image.recognition.dto.ToolClassificationDTO;
import com.image.recognition.dto.ToolDTO;
import com.image.recognition.exception.FileMissingException;
import com.image.recognition.exception.InvalidFileFormatException;
import com.image.recognition.service.ImageRecognitionService;
import com.image.recognition.service.ToolRecognitionService;

import javafx.util.Pair;

@RestController
@RequestMapping("/api")
public class ImageRecognitionController {

	private static final Logger LOGGER = Logger.getLogger(ImageRecognitionController.class.getName());
	private String[] allowedFileTypes = BaseImageLoader.ALLOWED_FORMATS;

	private static final Double RECOGNITION_THRESHOLD = 0.5;

	@Autowired
	private ImageRecognitionService imageRecognitionService;

	@Autowired
	private ToolRecognitionService toolRecognitionService;

	@GetMapping("/tools/detect")
	public ToolClassificationDTO getToolClassification(@RequestParam("imageFile") MultipartFile imageFile)
			throws IOException {
		if (imageFile != null && !imageFile.isEmpty()) {
			LOGGER.info("Type of image file: " + imageFile.getContentType());
			if (isFileFormatAllowed(imageFile)) {
				File file = convertMultipartFileToFile(imageFile);

				Map<ToolType, Double> toolPredictions = toolRecognitionService.detectTool(file);

				file.delete();

				Optional<Entry<ToolType, Double>> optClassifiedTool = toolRecognitionService
						.getToolWithHighestProbabilityFromPredictions(toolPredictions, RECOGNITION_THRESHOLD);
				
				ToolDTO toolDto;
				if (optClassifiedTool.isPresent()) {
					Entry<ToolType, Double> tool = optClassifiedTool.get();
					toolDto = new ToolDTO(tool.getKey(), tool.getValue());
				} else {
					toolDto = new ToolDTO(ToolType.valueOf("UNKNOWN"), 0.0);
				}
				
				ToolClassificationDTO classificationDto = new ToolClassificationDTO(toolDto, toolPredictions);
				/* Adding links to related endpoints and self */
				WebMvcLinkBuilder linkToClassificationLabels = linkTo(methodOn(this.getClass()).getToolLabels());
				Link selfRelLink = linkTo(ImageRecognitionController.class).withSelfRel();
				classificationDto.add(selfRelLink);
				classificationDto.add(linkToClassificationLabels.withRel("all-classification-labels"));

				return classificationDto;
			} else {
				throw new InvalidFileFormatException(
						"Invalid file format. Allowed formats: tif, jpg, png, jpeg, bmp, JPEG, JPG, TIF, PNG");
			}
		} else {
			throw new FileMissingException("No file provided in request.");
		}
	}
	
	@GetMapping("/tools/labels")
	public CollectionModel<ToolType> getToolLabels() {
		List<ToolType> toolLabels = Arrays.asList(ToolType.values());
		Link selfRelLink = linkTo(ImageRecognitionController.class).withSelfRel();
		CollectionModel<ToolType> labels = CollectionModel.of(toolLabels, selfRelLink);
		return labels;
	}

	@GetMapping("/animals/detect")
	public AnimalDTO getAnimalClassification(@RequestParam("imageFile") MultipartFile imageFile) throws IOException {
		if (imageFile != null && !imageFile.isEmpty()) {
			LOGGER.info("Type of image file: " + imageFile.getContentType());
			if (isFileFormatAllowed(imageFile)) {
				File file = convertMultipartFileToFile(imageFile);
				Pair<AnimalType, Double> typeOfAnimal = imageRecognitionService.detectAnimal(file, 0.5);

				file.delete();

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
				throw new InvalidFileFormatException(
						"Invalid file format. Allowed formats: tif, jpg, png, jpeg, bmp, JPEG, JPG, TIF, PNG");
			}
		} else {
			throw new FileMissingException("No file provided in request.");
		}

	}
	
	@GetMapping("/animals/labels")
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
		for (int i = 0; i < allowedFileTypes.length; i++) {
			if (fileType.equals(allowedFileTypes[i].toLowerCase())) {
				return true;
			}
		}
		return false;
	}
}
