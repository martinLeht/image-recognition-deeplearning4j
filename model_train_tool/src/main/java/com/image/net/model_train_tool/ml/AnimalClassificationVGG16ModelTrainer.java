package com.image.net.model_train_tool.ml;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Logger;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class AnimalClassificationVGG16ModelTrainer {

	private final static Logger LOGGER = Logger.getLogger(AnimalClassificationVGG16ModelTrainer.class.getName());
	
	/* Model data locations */
	public static String DATA_PATH;
	public static String TRAIN_FOLDER;
	public static String TEST_FOLDER;
	public static String SAVING_PATH;
    
	private ComputationGraph preTrainedNet;
	private ComputationGraph vgg16Transfer;
	
	
	/* Model Training parameters */
	private static final long seed = 12345;
	public static final Random RAND_NUM_GEN = new Random(seed);
	public static ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();
    public static final String[] ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS;
    public static BalancedPathFilter PATH_FILTER = new BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS, LABEL_GENERATOR_MAKER);
	
    
    /* Parameters for our training phase */
	private static final int EPOCH = 3;
    private static final int BATCH_SIZE = 16;
    private static final int TRAIN_SIZE = 85;
	private static final int OUTPUT_LABELS = 2;
	
	private static final int SAVING_INTERVAL = 100;
	    
    private static final String FREEZE_UNTIL_LAYER = "fc2";
    
    
    public AnimalClassificationVGG16ModelTrainer(String dataSavePath) {
    	DATA_PATH = dataSavePath;
    	TRAIN_FOLDER = DATA_PATH + "/train_both";
        TEST_FOLDER = DATA_PATH + "/test_both";
        SAVING_PATH = DATA_PATH + "/saved/modelIteration_";
	}
	
	public void initPreTrainedModelWithTransferLearning() throws IOException {
		ZooModel zooModel = VGG16.builder().build();
		
		preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		LOGGER.info(preTrainedNet.summary());
		
		if (preTrainedNet != null) {
			Nesterovs nesterovsUpdater = new Nesterovs();
			nesterovsUpdater.setLearningRate(5e-5);
			
			FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(nesterovsUpdater)
					.seed(seed).build();
			
			vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
					.fineTuneConfiguration(fineTuneConf)
					.setFeatureExtractor(FREEZE_UNTIL_LAYER)
					.removeVertexKeepConnections("predictions")
					.addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                .nIn(4096).nOut(OUTPUT_LABELS)
	                .weightInit(WeightInit.XAVIER)
	                .activation(Activation.SOFTMAX).build(), FREEZE_UNTIL_LAYER).build();
			vgg16Transfer.setListeners(new ScoreIterationListener(20));
			
			LOGGER.info(vgg16Transfer.summary());
		}
	}
	
	public void trainModel() throws IOException {		
		if (preTrainedNet != null) {
			 // Define the File Paths
	        File trainData = new File(TRAIN_FOLDER);
	        File testData = new File(TEST_FOLDER);
	        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);
	        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

	        InputSplit[] sample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
	        DataSetIterator trainIterator = getDataSetIterator(sample[0]);
	        DataSetIterator devIterator = getDataSetIterator(sample[1]);
			DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0]);
			
	        int iEpoch = 0;
	        int i = 0;
	        while (iEpoch < EPOCH) {
	            while (trainIterator.hasNext()) {
	                DataSet trained = trainIterator.next();
	                vgg16Transfer.fit(trained);
	                if (i % SAVING_INTERVAL == 0 && i != 0) {
	                    ModelSerializer.writeModel(vgg16Transfer, new File(SAVING_PATH + i + "_epoch_" + iEpoch + ".zip"), false);
	                    evalOn(vgg16Transfer, devIterator, i);
	                }
	                i++;
	            }
	            trainIterator.reset();
	            iEpoch++;

	            evalOn(vgg16Transfer, testIterator, iEpoch);
	        }
		}
	
	}
	
	public static void evalOn(ComputationGraph vgg16Transfer, DataSetIterator testIterator, int iEpoch) throws IOException {
        LOGGER.info("Evaluate model at iteration " + iEpoch + " ....");
        Evaluation eval = vgg16Transfer.evaluate(testIterator);
        LOGGER.info(eval.stats());
        testIterator.reset();
    }

    public static DataSetIterator getDataSetIterator(InputSplit sample) throws IOException {
        ImageRecordReader imageRecordReader = new ImageRecordReader(224, 224, 3, LABEL_GENERATOR_MAKER);
        imageRecordReader.initialize(sample);

        DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, OUTPUT_LABELS);
        iterator.setPreProcessor(new VGG16ImagePreProcessor());
        return iterator;
    }

}
