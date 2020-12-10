package com.image.net.model_train_tool.ml;

public enum ToolType {
	GASOLINE_CAN("Gasoline Can"),
    HAMMER("Hammer"),
    PEBBELS("Pebbels"),
    PILERS("Pilers"),
    ROPE("Rope"),
    SCREW_DRIVER("Screw Driver"),
    TOOLBOX("Toolbox"),
    WRENCH("Wrench"),
    UNKNOWN("Unknown");
	
	private String type;
	
	ToolType(String type) {
		this.type = type;
	}
	
	public String getType() {
		return type;
	}
}
