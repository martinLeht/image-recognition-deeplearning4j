package com.image.net.model_train_tool;

import com.image.net.model_train_tool.ml.RunnableImageNetVGG16;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;



/**
 * A commandline program to train a VGG16 based Image net model.
 * Syntax: java -cp "pathToPicocliJar:pathToModelTrainerToolJar" ModelTrainerTool	
 * Run program with default or specified data saving location e.g.:
 * 		java -cp "picocli-4.5.2.jar:ModelTrainerTool.jar" ModelTrainerTool	 
 * 		java -cp "picocli-4.5.2.jar:ModelTrainerTool.jar" ModelTrainerTool --path C:/users/myUser/files/recognition_model
 * @author SaitamasPC
 *
 */
@Command(name = "ModelTrainerTool", 
		description = "Trains and saves a image recognition model based on VGG16 to desired location (default location = userPath/image_recognition_model)."
					+ " Option -p, -P or -path can be used to specify saving location.")
public class ModelTrainerTool {
	
	/* *
	 * Option to set custom path to model data save location
	 * Default: userPath/image_recognition_model/
	 * Can be specified with option: -p, -P, --path
	 * E.g.: java -cp "picocli-4.5.2.jar:ModelTrainerTool.jar" ModelTrainerTool --path C:/users/myUser/files/recognition_model
	 * */
	@Option(names = {"-p", "-P", "--path"})
	public static String dataSaveLocation = System.getProperty("user.home") + "/image_recognition_model";
	
	public static void main(String[] args) {
		int exitCode = new CommandLine(new RunnableImageNetVGG16(dataSaveLocation)).execute(args);
		System.exit(exitCode);
	}
}
