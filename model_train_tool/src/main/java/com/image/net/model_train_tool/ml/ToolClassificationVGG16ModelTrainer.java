package com.image.net.model_train_tool.ml;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.profiler.ProfilerConfig;

public class ToolClassificationVGG16ModelTrainer implements IModelTrainer {

	enum DataSetType {
		TRAIN, VALID, TEST
	}

	/* Logging */
	private final static Logger LOGGER = Logger.getLogger(ToolClassificationVGG16ModelTrainer.class.getName());
	FileHandler modelTrainerLogFileHandler;
	public static String LOG_PATH;

	/* Model data locations */
	public static String DATA_PATH;
	public static String TRAIN_FOLDER;
	public static String TEST_FOLDER;
	public static String MODEL_VGG16_SAVING_PATH;
	// public static String MULTILAYER_SAVING_PATH;

	/* For pretrained and transfer learning model */
	private ComputationGraph vgg16Transfer;

	/* Model Training parameters */
	private static final long seed = 12345;
	private static final long seed2 = 54321;
	public static final Random RNG = new Random(seed);
	public static final Random RNG_2 = new Random(seed2);
	public static ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();
	public static final String[] ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS;
	public static BalancedPathFilter PATH_FILTER = new BalancedPathFilter(RNG, ALLOWED_FORMATS, LABEL_GENERATOR_MAKER);

	/* Input Tensor Dimensions */
	private static final int HEIGHT = 224;
	private static final int WIDTH = 224;
	private static final int CHANNELS = 3;

	/* Parameters for training phase */
	private static final int EPOCH = 3;
	private static final int BATCH_SIZE = 16;
	private static final int TRAIN_SIZE = 80;
	private static final int NUM_OUTPUT_LABELS = 8;

	/* Interval for how often model is evaluated while training */
	private static final int EVAL_INTERVAL = 10;

	/* Layers freezed until this layer, weights are kept fixed while training */
	private static final String FREEZE_UNTIL_LAYER = "fc2";

	public ToolClassificationVGG16ModelTrainer(String dataSavePath) {
		DATA_PATH = dataSavePath;
		LOG_PATH = DATA_PATH + "/log";
		TRAIN_FOLDER = DATA_PATH + "/tool_data/train_all";
		TEST_FOLDER = DATA_PATH + "/tool_data/test_all";
		MODEL_VGG16_SAVING_PATH = DATA_PATH + "/saved";

		/* Create directories if they dont exist */
		File modelSavingDir = new File(MODEL_VGG16_SAVING_PATH);
		if (!modelSavingDir.exists()) {
			modelSavingDir.mkdir();
		}

		File logDir = new File(LOG_PATH);
		if (!logDir.exists()) {
			logDir.mkdir();
		}

		/* Setting up the logger */
		try {
			modelTrainerLogFileHandler = new FileHandler(LOG_PATH + "/model_trainer_app.log");
			LOGGER.addHandler(modelTrainerLogFileHandler);

			// Print the LogRecord in a human readable format.
			SimpleFormatter formatter = new SimpleFormatter();
			modelTrainerLogFileHandler.setFormatter(formatter);
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	@Override
	public void initPreTrainedModelWithTransferLearning() throws IOException {

		ZooModel zooModel = VGG16.builder().build();
		ComputationGraph preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

		if (preTrainedNet != null) {
			LOGGER.info(preTrainedNet.summary());

			FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.updater(new Nesterovs(5e-4, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
					.seed(seed)
					.build();

			/*
			 * no new layers, except output
			 */
			vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet).fineTuneConfiguration(fineTuneConf)
					.setFeatureExtractor(FREEZE_UNTIL_LAYER)
					.removeVertexKeepConnections("predictions")
					.addLayer("predictions",
							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.nIn(4096)
									.nOut(NUM_OUTPUT_LABELS)
									.weightInit(WeightInit.XAVIER)
									.activation(Activation.SOFTMAX)
								.build(),
							FREEZE_UNTIL_LAYER)
					.build();

			vgg16Transfer.setListeners(new ScoreIterationListener(5));

			LOGGER.info(vgg16Transfer.summary());
		} else {
			LOGGER.info("Something went wrong with the initialization of the model.");
		}
	}

	private Layer createOutputLayer(int nIn, int nOut) {
		return new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					.nIn(nIn)
					.nOut(nOut)
					.activation(Activation.SOFTMAX)
				.build();
	}
	
	
	@Override
	public void trainPretrainedModel() throws IOException {
		if (vgg16Transfer != null) {

			// Define the File Paths
			File trainData = new File(TRAIN_FOLDER);
			File testData = new File(TEST_FOLDER);
			FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RNG);
			FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RNG);

			InputSplit[] trainDataSample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
			DataSetIterator trainIterator = getDataSetIterator(trainDataSample[0], true);
			DataSetIterator validIterator = getDataSetIterator(trainDataSample[1], false);
			DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0], false);

			int iEpoch = 0;
			int i = 0;
			while (iEpoch < EPOCH) {
				while (trainIterator.hasNext()) {
					DataSet trained = trainIterator.next();
					vgg16Transfer.fit(trained);
					if (i % EVAL_INTERVAL == 0 && i != 0) {
						evalModelOn(validIterator, i);
					}
					i++;
				}
				trainIterator.reset();
				iEpoch++;
				evalModelOn(testIterator, iEpoch);
			}
			ModelSerializer.writeModel(vgg16Transfer,
					new File(MODEL_VGG16_SAVING_PATH + "/tool_classification_model_vgg16.zip"), false);
		} else {
			LOGGER.info("Model needs to be initialized before training. "
					+ "Run ToolClassificationVGG16ModelTrainer.initPreTrainedModelWithTransferLearning() and try training again.");
		}

	}

	private void evalModelOn(DataSetIterator iterator, int iter) throws IOException {
		LOGGER.info("Evaluate model at iteration " + iter + " ....");
		Evaluation eval = vgg16Transfer.evaluate(iterator);
		LOGGER.info(eval.stats());
		iterator.reset();
	}

	public DataSetIterator getDataSetIterator(InputSplit sample, boolean withDataAugmentation) throws IOException {
		ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, LABEL_GENERATOR_MAKER);
		imageRecordReader.initialize(sample);
		if (withDataAugmentation) {
			List<ImageTransform> transforms = dataAugmentation();
			for (ImageTransform transform : transforms) {
				imageRecordReader.initialize(sample, transform);
			}
		}

		DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, NUM_OUTPUT_LABELS);
		iterator.setPreProcessor(new VGG16ImagePreProcessor());
		return iterator;
	}

	private List<ImageTransform> dataAugmentation() {
		ImageTransform flipTransform1 = new FlipImageTransform(RNG);
		ImageTransform flipTransform2 = new FlipImageTransform(RNG_2);
		ImageTransform warpTransform = new WarpImageTransform(RNG, 42);
		ImageTransform rotationTransform = new RotateImageTransform(RNG, 90);

		return Arrays.asList(new ImageTransform[] { flipTransform1, warpTransform, flipTransform2, rotationTransform });
	}

}
