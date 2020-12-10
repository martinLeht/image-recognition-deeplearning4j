package com.image.net.model_train_tool.ml;

import java.io.IOException;

public interface IModelTrainer {
	/*
	 * Initializes the pre-trained ZooModel
	 * Transfers learning from pre-trained model to our own VGG16 model
	 * Handles Fine tune configuration of our model
	 */
	public void initPreTrainedModelWithTransferLearning() throws IOException;
	
	/*
	 * Initializes a MultiLayer CNN
	 */
	public void init() throws IOException;
	
	/*
	 * Performas the training of our model
	 * Saves and evaluates the model while training according to SAVING_INTERVAL 
	 * Finally evaluates and saves when converged through last epoch
	 */
	public void trainPretrainedModel() throws IOException;
	
	/*
	 * Performas the training of our model
	 * Saves and evaluates the model while training according to SAVING_INTERVAL 
	 * Finally evaluates and saves when converged through last epoch
	 */
	public void trainModel() throws IOException;
}
