package HomeWork2;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Instance;

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
     * @param instances
     */
    public void buildTree(Instances instances) {
    	
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
     * Calculates, for set of instances, their probabilities  in
     * preparation to calculate purity (entropy),
     * for every possible attribute
     *
     * @param instances set of instances (in certain node, probably)
     * @return probabilities of positive class (1) for each possible attribute
     */
    public double[] calcPositiveProbabilities(Instances instances){
    	int numOfAttributes = instances.numAttributes();
    	int numOfInstances = instances.numInstances();
    	int classIndex = instances.classIndex();
    	int numOfYes = 0;
    	double[] probabilities = new double[numOfAttributes];
    	
    	// runs on all possible attributes, as long as it's not the classIndex
    	// and for each attribute sums the num of "yes"s (1)
    	for (int i = 0; i < numOfAttributes; i++){
    		if (i != classIndex){
    			for (int j = 0; j < numOfInstances; j++){
        			if (instances.instance(j).classValue() == 1){
        				numOfYes++;
        			}
        		}
    			// after summing up all the "yes"s for every instance
    			// for the given attribute, calculates the probability
    			// and stores in the array
    			probabilities[i] = numOfYes / (double)numOfInstances;
    			numOfYes = 0; // zeros sum of "yes"s before next iteration
    		}
    	}
    	return probabilities;
    }
    
    
    /**
     * Calculates the entropy of a random variable where all the probabilities 
     * of all of the possible values it can take are given as input.
     * 
     * @param probabilities - A set of probabilities
     * @return The entropy 
     */
    public double calcEntropy(double[] probabilities){
    	int numOfInstances = probabilities.length;
    	double entropy;
    	double probOfCurrEvent;
    	double tempCalcSi0; // these are the NOs
    	double tempCalcSi1; //these are the YESs - what is calculated directly in the given array
    	
    	// calculates 
    	for (int i = 0; i < numOfInstances; i++){
    		tempCalcSi1 = probabilities[i];
    		entropy = ((tempCalcSi1 * (Math.log(tempCalcSi1) / Math.log(2.0))) 
    					+ (probOfNo * (Math.log(probOfNo) / Math.log(2.0))));
    	}
    	
    	
    	//// calculates prob (using log tricks to assure base 2) according to given formula
		//currAttributeProb = - ((probOfYes * (Math.log(probOfYes) / Math.log(2.0))) 
			//	+ (probOfNo * (Math.log(probOfNo) / Math.log(2.0))));
    	
    	
    	
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
    	// xj is the attribute at index j (attributeIndex)
    	int numValues = instances.attribute(attributeIndex).numValues();
    	int numInstances = instances.numInstances();
    	// number of instances for which attribute value at (j) = val(f)
    	int numInstancesWithCurValue = 0;
    	// Positives: number of instances for which (attVal=f) and (Y = 1)
    	int numInstanceswithFAndPos = 0;
    	// Negatives: number of instances for which (attVal=f) and (Y = 0)
    	int numInstanceswithFAndNeg = 0;
    	double posE, negE;
    	double chiSquare = 0;
    	
    	// calculate number of positive and negative instances
    	int numPositive = 0;
    	int numNegative = 0;
    	for (int i = 0; i < numInstances; i++) {
    		Instance curInstance = instances.instance(i);
    		if (curInstance.classValue() == 1) {
    			numPositive ++;
    		} else {
    			numNegative ++;
    		}
    		
    	}
    	double Py0 = numNegative / (double) numInstances;
    	double Py1 = numPositive / (double) numInstances;
    	
    	// going over all possible values
    	for (int f = 0; f < numValues; f++) {
			double tempCalc = 0;
			// calculating number of instances which j attribute value is f
			for (int i = 0; i < numInstances; i++) {
				Instance curInstance = instances.instance(i);
				if (curInstance.attribute(attributeIndex).value(f) == 
						instances.attribute(attributeIndex).value(f)){
					numInstancesWithCurValue++;
					if (curInstance.classValue() == 1) {
						numInstanceswithFAndPos++;
					} else {
						numInstanceswithFAndNeg++;
					}
				}
			}
			posE = numInstancesWithCurValue * Py1;
			negE = numInstancesWithCurValue * Py0;
			
			// adding calculation to chi square
			tempCalc = (Math.pow((numInstanceswithFAndPos - posE), 2) / posE) + 
					(Math.pow((numInstanceswithFAndNeg - negE), 2) / negE);
			chiSquare += tempCalc;
			numInstancesWithCurValue = 0;
			numInstanceswithFAndPos = 0;
			numInstanceswithFAndNeg = 0;
		}
    	
    	return chiSquare;
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
