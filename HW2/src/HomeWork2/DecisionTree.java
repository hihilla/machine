package HomeWork2;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class BasicRule {
    int attributeIndex;
	int attributeValue;
}

class Rule {
   	List<BasicRule> basicRule;
   	double returnValue;
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
   	Rule nodeRule = new Rule();

}

public class DecisionTree implements Classifier {
	private Node rootNode;
	public enum PruningMode {None, Chi, Rule};
	private PruningMode m_pruningMode;
   	Instances validationSet;
   	private List<Rule> rules = new ArrayList<Rule>();

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}
	
	public void setValidation(Instances validation) {
		validationSet = validation;
	}
    
    @Override
	public double classifyInstance(Instance instance) {
		//TODO: implement this method
    	return 0;
	}
    
    /**
     * Builds the decision tree on given data set using either a recursive or 
     * queue algorithm.
     * @param instance
     */
    public void buildTree(Instance instance) {
    	
    }
    
    /**
     * Calculate the average on a given instances set (could be the training, 
     * test or validation set). The average error is the total number of 
     * classification mistakes on the input instances set and divides that by 
     * the number of instances in the input set.
     * @param instance
     * @return Average error 
     */
    public double calcAvgError(Instance instance){
    	return 0;
    }
    
    /**
     * calculates the information gain of splitting the input data according to 
     * the attribute.
     * @param instance
     * @return The information gain 
     */
    public double calcInfoGain(Instance instance){
    	return 0;
    }
    
    /**
     * Calculates the entropy of a random variable where all the probabilities 
     * of all of the possible values it can take are given as input.
     * @param probabilities - A set of probabilities
     * @return The entropy 
     */
    public double calcEntropy(double[] probabilities){
    	return 0;
    }
    
    /**
     * Calculates the chi square statistic of splitting the data according to 
     * this attribute as learned in class.
     * @param instances - a subset of the training data
     * @param attributeIndex
     * @return The chi square score
     */
    public double calcChiSquare(Instances instances, int attributeIndex){
    	return 0;
    }
    
    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
