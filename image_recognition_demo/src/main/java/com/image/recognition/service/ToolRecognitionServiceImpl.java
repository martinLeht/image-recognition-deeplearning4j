package com.image.recognition.service;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.logging.Logger;

import javax.annotation.PostConstruct;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.springframework.stereotype.Service;

import com.image.net.model_train_tool.ml.ToolType;


@Service
public class ToolRecognitionServiceImpl implements ToolRecognitionService {
	
private final static Logger LOGGER = Logger.getLogger(ToolRecognitionServiceImpl.class.getName());
	
	private static final String TRAINED_MODEL_PATH = "src/main/resources/saved/tool_modelvgg16.zip";
    private static final int OUTPUT_LABELS = 8;
    
    private static ComputationGraph computationGraph;
    private DataNormalization scaler;
    private NativeImageLoader imageLoader;
    
    @PostConstruct
    public void init() {
    	scaler = new VGG16ImagePreProcessor();
    	/* VGG16 model input dimensions */
    	imageLoader = new NativeImageLoader(224, 224, 3);
    }
	
	@Override
	public Map<ToolType, Double> detectTool(File file) throws IOException {
		if (computationGraph == null) {
			computationGraph = loadModel();
        }
		
		computationGraph.init();
        LOGGER.info(computationGraph.summary());
        INDArray image = imageLoader.asMatrix(new FileInputStream(file));
        scaler.transform(image);
        
        INDArray output = computationGraph.outputSingle(false, image);
        return getPredictions(output);
	}
	
	private ComputationGraph loadModel() throws IOException {
		computationGraph = ModelSerializer.restoreComputationGraph(new File(TRAINED_MODEL_PATH));
        return computationGraph;
    }
	
	private Map<ToolType, Double> getPredictions(INDArray output) {
		Map<ToolType, Double> toolPredictions = new HashMap<>();
		for(int i = 0; i < OUTPUT_LABELS ; i++) {
			Double prediction = output.getDouble(i);
			switch(i) {
				case 0:
					toolPredictions.put(ToolType.valueOf("GASOLINE_CAN"), prediction);
					break;
					
				case 1:
					toolPredictions.put(ToolType.valueOf("HAMMER"), prediction);
					break;
					
				case 2:
					toolPredictions.put(ToolType.valueOf("PILERS"), prediction);
					break;
					
				case 3:
					toolPredictions.put(ToolType.valueOf("ROPE"), prediction);
					break;
					
				case 4:
					toolPredictions.put(ToolType.valueOf("SCREW_DRIVER"), prediction);
					break;
					
				case 5:
					toolPredictions.put(ToolType.valueOf("TOOLBOX"), prediction);
					break;
					
				case 6:
					toolPredictions.put(ToolType.valueOf("WRENCH"), prediction);
					break;
					
				case 7:
					toolPredictions.put(ToolType.valueOf("PEBBELS"), prediction);
					break;
					
				default:
					break;
			}
		}
		return toolPredictions;
	}

	@Override
	public Optional<Entry<ToolType, Double>> getToolWithHighestProbabilityFromPredictions(Map<ToolType, Double> predictions, Double threshold) {
		
		Optional<Entry<ToolType, Double>> optToolWithHighestProbability = predictions.entrySet()
			.stream()
			.filter(e -> e.getValue() > threshold)
			.max(Comparator.comparingDouble(Entry::getValue));
		return optToolWithHighestProbability;
	}
	
}
