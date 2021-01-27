package com.image.net.model_train_tool;

import java.io.IOException;

import com.image.net.model_train_tool.ml.IModelTrainer;
import com.image.net.model_train_tool.ml.ToolClassificationVGG16ModelTrainer;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;


/**
 * A command line tool to train a VGG16 based Image classification model.
 * <br>Should be packaged as executable JAR -file containing all dependencies with "mvn package" command.
 * <p>
 * Syntax: java -jar target/model-train-tool-jar-with-dependencies.jar
 * <p>
 * Run the JAR (mvn assembly with all dependencies) from project root:
 * <br>- With default base location for dataset and model saving:
 * 		<br>java -jar target/model-train-tool-jar-with-dependencies.jar
 * <br>- With specified  base location for dataset and model saving:  
 * 		<br>java -jar target/model-train-tool-jar-with-dependencies.jar --path C:/users/myUser/files/recognition_model
 * @author Martin Lehtomaa
 *
 */
@Command(name = "ModelTrainerTool", 
		description = "Trains and saves a Mechanical tool classification model to desired location."
				+ " <br>Default location: C:/users/userPath/image_recognition_model."
				+ " <br>Optional argument: -p, -P or --path can be used to specify location.")
public class ModelTrainerTool implements Runnable {
	
	/**
	 * Option to set custom path to dataset and model data saving location
	 * Default: userPath/image_recognition_model/
	 * Can be specified with option: -p, -P, --path
	 * */
	@Option(names = {"-p", "-P", "--path"})
	public static String dataSaveLocation = System.getProperty("user.home") + "\\image_recognition_model";
	
	public static void main(String[] args) {
		int exitCode = new CommandLine(new ModelTrainerTool()).execute(args);
		System.exit(exitCode);
	}

	@Override
	public void run() {
		IModelTrainer modelTrainer = new ToolClassificationVGG16ModelTrainer(dataSaveLocation);
		try {			
			modelTrainer.initPreTrainedModelWithTransferLearning();
			modelTrainer.trainPretrainedModel();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
