package com.image.net.model_train_tool;

import java.io.IOException;

import com.image.net.model_train_tool.ml.IModelTrainer;
import com.image.net.model_train_tool.ml.AnimalClassificationVGG16ModelTrainer;
import com.image.net.model_train_tool.ml.ToolClassificationVGG16ModelTrainer;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;



/**
 * A commandline program to train a VGG16 based Image net model.
 * Syntax: java -cp "pathToModelTrainerToolJar" fully.Qualified.Package.Path.ClassName
 * Run program as JAR (mvn assembly with all dependencies) from project root:
 * - With default data saving location:
 * 		java -cp target/model-train-tool-jar-with-dependencies.jar com.image.net.model_train_tool.ModelTrainerTool
 * - With specified data saving location  
 * 		java -cp target/model-train-tool-jar-with-dependencies.jar com.image.net.model_train_tool.ModelTrainerTool --path C:/users/myUser/files/recognition_model
 * @author SaitamasPC
 *
 */
@Command(name = "ModelTrainerTool", 
		description = "Trains and saves a image recognition model based on VGG16 to desired location (default location = userPath/image_recognition_model)."
					+ " Option -p, -P or -path can be used to specify saving location.")
public class ModelTrainerTool implements Runnable {
	
	/**
	 * Option to set custom path to model data save location
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
