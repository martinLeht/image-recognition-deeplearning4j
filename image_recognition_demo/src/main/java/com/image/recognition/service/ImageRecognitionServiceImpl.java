package com.image.recognition.service;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.logging.Logger;

import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.springframework.stereotype.Service;

import com.image.net.model_train_tool.ml.AnimalType;
import com.image.net.model_train_tool.ml.RunnableImageNetVGG16;

import javafx.util.Pair;

@Service
public class ImageRecognitionServiceImpl implements ImageRecognitionService {
	
	private final static Logger LOGGER = Logger.getLogger(ImageRecognitionServiceImpl.class.getName());
	
	private static final String TRAINED_PATH_MODEL = "src/main/resources/saved/modelIteration_3900_epoch_2.zip";
			//RunnableImageNetVGG16.DATA_PATH + "/model.zip";
    private static ComputationGraph computationGraph;
	
	public Pair<AnimalType, Double> detectAnimal(File file, Double threshold) throws IOException {
		if (computationGraph == null) {
            computationGraph = loadModel();
        }
		
		computationGraph.init();
        LOGGER.info(computationGraph.summary());
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(new FileInputStream(file));
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        INDArray output = computationGraph.outputSingle(false, image);
        if (output.getDouble(0) > threshold) {
            return new Pair<>(AnimalType.valueOf("CAT"), output.getDouble(0));
        } else if (output.getDouble(1) > threshold) {
            return new Pair<>(AnimalType.valueOf("DOG"), output.getDouble(1));
        } else {
            return new Pair<>(AnimalType.valueOf("UNKNOWN"), -1.0);
        }
	}
	
	private ComputationGraph loadModel() throws IOException {
        computationGraph = ModelSerializer.restoreComputationGraph(new File(TRAINED_PATH_MODEL));
        return computationGraph;
    }
	
	private void runOnTestSet() throws IOException {
        ComputationGraph computationGraph = loadModel();
        File trainData = new File(RunnableImageNetVGG16.TEST_FOLDER);
        FileSplit test = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RunnableImageNetVGG16.RAND_NUM_GEN);
        InputSplit inputSplit = test.sample(RunnableImageNetVGG16.PATH_FILTER, 100, 0)[0];
        DataSetIterator dataSetIterator = RunnableImageNetVGG16.getDataSetIterator(inputSplit);
        RunnableImageNetVGG16.evalOn(computationGraph, dataSetIterator, 1);
    }

    private void runOnDevSet() throws IOException {
        ComputationGraph computationGraph = loadModel();
        File trainData = new File(RunnableImageNetVGG16.TRAIN_FOLDER);
        FileSplit test = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RunnableImageNetVGG16.RAND_NUM_GEN);
        InputSplit inputSplit = test.sample(RunnableImageNetVGG16.PATH_FILTER, 15, 85)[0];
        DataSetIterator dataSetIterator = RunnableImageNetVGG16.getDataSetIterator(inputSplit);
        RunnableImageNetVGG16.evalOn(computationGraph, dataSetIterator, 1);
    }
}
