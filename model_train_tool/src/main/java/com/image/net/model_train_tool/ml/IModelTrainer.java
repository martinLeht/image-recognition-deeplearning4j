package com.image.net.model_train_tool.ml;

import java.io.IOException;

public interface IModelTrainer {
	/**
	 * Initializes the pre-trained ZooModel.
	 * <br>Transfers learning from pre-trained model to the new VGG16 based model.
	 * <br>Handles Fine-tune configuration of the new model.
	 */
	public void initPreTrainedModelWithTransferLearning() throws IOException;

	/**
	 * Performs the training of the model.
	 * <br>IMPORTANT: Run method IModelTrainer.initPreTrainedModelWithTransferLearning() before training!
	 * <p>
	 * Pipeline:
	 * <br>- Saves and evaluates the model while training according to SAVING_INTERVAL 
	 * <br>- Finally evaluates and saves the model as .zip file when converged through last epoch
	 */
	public void trainPretrainedModel() throws IOException;
	
	
	/**
	 * Initializes a MultiLayer CNN from scratch.
	 */
	//public void initFromScratch() throws IOException;
	
	/**
	 * Performs the training of the model.
	 * <br>IMPORTANT: Run method IModelTrainer.initFromScratch() before training!
	 * <p>
	 * Pipeline:
	 * <br>- Saves and evaluates the model while training according to SAVING_INTERVAL 
	 * <br>- Finally evaluates and saves the model as .zip file when converged through last epoch
	 */
	//public void trainModelFromScratch() throws IOException;
	
}
