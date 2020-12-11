package com.image.net.model_train_tool.ml;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.Xception;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ToolClassificationVGG16ModelTrainer implements IModelTrainer {

	private final static Logger LOGGER = Logger.getLogger(ToolClassificationVGG16ModelTrainer.class.getName());

	/* Model data locations */
	public static String DATA_PATH;
	public static String TRAIN_FOLDER;
	public static String TEST_FOLDER;
	public static String MODEL_VGG16_SAVING_PATH;
	public static String MULTILAYER_SAVING_PATH;

	private ComputationGraph preTrainedNet;
	private ComputationGraph vgg16Transfer;

	private MultiLayerNetwork multiLayerNetwork;

	/* Model Training parameters */
	private static final long seed = 12345;
	public static final Random RAND_NUM_GEN = new Random(seed);
	public static ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();
	public static final String[] ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS;
	public static BalancedPathFilter PATH_FILTER = new BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS,
			LABEL_GENERATOR_MAKER);

	/* Input Data Dimensions */
	private static final int HEIGHT = 224;
	private static final int WIDTH = 224;
	private static final int CHANNELS = 3;

	/* Parameters for our training phase */
	private static final int EPOCH = 5;
	private static final int BATCH_SIZE = 16;
	private static final int TRAIN_SIZE = 80;
	private static final int OUTPUT_LABELS = 8;

	private static final int EVAL_INTERVAL = 20;

	private static final String FREEZE_UNTIL_LAYER = "fc2";

	public ToolClassificationVGG16ModelTrainer(String dataSavePath) {
		DATA_PATH = dataSavePath;
		TRAIN_FOLDER = DATA_PATH + "/tool_data/train_all";
		TEST_FOLDER = DATA_PATH + "/tool_data/test_all";
		MODEL_VGG16_SAVING_PATH = DATA_PATH + "/tool_data/saved/modelvgg16";
		MULTILAYER_SAVING_PATH = DATA_PATH + "/tool_data/saved/multilayer_model";
	}

	@Override
	public void initPreTrainedModelWithTransferLearning() throws IOException {
		
		ZooModel zooModel = VGG16.builder().build();
		//ZooModel zooModel = Xception.builder().build();

		preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		LOGGER.info(preTrainedNet.summary());

		if (preTrainedNet != null) {
			Nesterovs nesterovsUpdater = new Nesterovs();
			nesterovsUpdater.setLearningRate(5e-5);

			FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.updater(nesterovsUpdater)
					.seed(seed)
					.build();

			vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
					.fineTuneConfiguration(fineTuneConf)
					.setFeatureExtractor(FREEZE_UNTIL_LAYER)
					.removeVertexKeepConnections("predictions")
					.addLayer("predictions",
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.nIn(4096)
									.nOut(OUTPUT_LABELS)
									.weightInit(WeightInit.XAVIER)
									.activation(Activation.SOFTMAX)
									.build(),
							FREEZE_UNTIL_LAYER)
					.build();
			vgg16Transfer.setListeners(new ScoreIterationListener(20));

			LOGGER.info(vgg16Transfer.summary());
		}
	}

	@Override
	public void init() throws IOException {
		Nesterovs nesterovsUpdater = new Nesterovs();
		nesterovsUpdater.setLearningRate(5e-5);
		MultiLayerConfiguration multiLayerConf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.l2(0.005)
				.activation(Activation.RELU)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(nesterovsUpdater)
				.list()
				.layer(0, convInit("cnn1", CHANNELS, 50, new int[] { 5, 5 }, new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(1, maxPool("maxpool1", new int[] { 2, 2 }))
				.layer(2, conv5x5("cnn2", 100, new int[] { 5, 5 }, new int[] { 1, 1 }, 0))
				.layer(3, maxPool("maxool2", new int[] { 2, 2 }))
				.layer(4, new DenseLayer.Builder()
							.nOut(500)
							.build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
							.nOut(OUTPUT_LABELS)
							.activation(Activation.SOFTMAX)
							.build())
				.setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
				.backpropType(BackpropType.Standard)
				.build();

		multiLayerNetwork = new MultiLayerNetwork(multiLayerConf);
	}

	private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad,
			double bias) {
		return new ConvolutionLayer.Builder(kernel, stride, pad)
				.name(name)
				.nIn(in)
				.nOut(out)
				.biasInit(bias)
				.build();
	}

	private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 5, 5 }, stride, pad)
				.name(name)
				.nOut(out)
				.biasInit(bias)
				.build();
	}

	private SubsamplingLayer maxPool(String name, int[] kernel) {
		return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 })
				.name(name)
				.build();
	}

	public void trainPretrainedModel() throws IOException {
		if (preTrainedNet != null) {
			// Define the File Paths
			File trainData = new File(TRAIN_FOLDER);
			File testData = new File(TEST_FOLDER);
			FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);
			FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

			InputSplit[] sample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
			DataSetIterator trainIterator = getDataSetIterator(sample[0], true, true);
			DataSetIterator devIterator = getDataSetIterator(sample[1], false, true);
			DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0], false, true);

			int iEpoch = 0;
			int i = 0;
			while (iEpoch < EPOCH) {
				while (trainIterator.hasNext()) {
					DataSet trained = trainIterator.next();
					vgg16Transfer.fit(trained);
					if (i % EVAL_INTERVAL == 0 && i != 0) {
						evalPretrainedModelOn(devIterator, i);
					}
					i++;
				}
				trainIterator.reset();
				iEpoch++;
			}
			ModelSerializer.writeModel(vgg16Transfer, new File(MODEL_VGG16_SAVING_PATH + ".zip"),
					false);
			evalPretrainedModelOn(testIterator, iEpoch);
		}

	}

	public void trainModel() throws IOException {
		if (multiLayerNetwork != null) {
			// Define data sets
			File trainData = new File(TRAIN_FOLDER);
			File testData = new File(TEST_FOLDER);
			FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);
			FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

			InputSplit[] sample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
			DataSetIterator trainIterator = getDataSetIterator(sample[0], true, false);
			DataSetIterator devIterator = getDataSetIterator(sample[1], false, false);
			DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0], false, false);
			
			int iEpoch = 0;
			int i = 0;
			while (iEpoch < EPOCH) {
				while (trainIterator.hasNext()) {
					DataSet trained = trainIterator.next();
					multiLayerNetwork.fit(trained);
					
					if (i % EVAL_INTERVAL == 0 && i != 0) {
						//multiLayerNetwork.save(new File(MULTILAYER_SAVING_PATH + "_iter_" + i + "_epoch_" + iEpoch + ".zip"), false);
						evalMultiLayerModelOn(devIterator, i);
					}
					i++;
					
				}
				trainIterator.reset();
				//multiLayerNetwork.save(new File(MULTILAYER_SAVING_PATH + "_iter_" + i + "_epoch_" + iEpoch + ".zip"), false);
				iEpoch++;
				//evalMultiLayerModelOn(testIterator, iEpoch);
			}
			multiLayerNetwork.save(new File(MULTILAYER_SAVING_PATH + ".zip"), false);
			evalMultiLayerModelOn(testIterator, iEpoch);
		}

	}

	private void evalPretrainedModelOn(DataSetIterator testIterator, int iEpoch)
			throws IOException {
		LOGGER.info("Evaluate model at iteration " + iEpoch + " ....");
		Evaluation eval = vgg16Transfer.evaluate(testIterator);
		LOGGER.info(eval.stats());
		testIterator.reset();
	}
	
	private void evalMultiLayerModelOn(DataSetIterator testIterator, int iEpoch)
			throws IOException {
		LOGGER.info("Evaluate model at iteration " + iEpoch + " ....");
		Evaluation eval = multiLayerNetwork.evaluate(testIterator);
		LOGGER.info(eval.stats());
		testIterator.reset();
	}

	public DataSetIterator getDataSetIterator(InputSplit sample, boolean withDataAugmentation,
			boolean forVGG16) throws IOException {
		ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, LABEL_GENERATOR_MAKER);
		if (withDataAugmentation) {
			List<ImageTransform> transforms = dataAugmentation();
			for (ImageTransform transform : transforms) {
				imageRecordReader.initialize(sample, transform);
			}
		} else {
			imageRecordReader.initialize(sample);
		}

		DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, OUTPUT_LABELS);
		if (forVGG16) {
			iterator.setPreProcessor(new VGG16ImagePreProcessor());
		} else {
			DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
			scaler.fit(iterator);
			iterator.setPreProcessor(scaler);
		}
		return iterator;
	}
	
	private List<ImageTransform> dataAugmentation() {
		ImageTransform flipTransform1 = new FlipImageTransform(RAND_NUM_GEN);
		ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
		ImageTransform warpTransform = new WarpImageTransform(RAND_NUM_GEN, 42);
		return Arrays.asList(new ImageTransform[] { flipTransform1, warpTransform, flipTransform2 });
	}

}
