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

	public enum PruningMode {
		None, Chi, Rule
	};

	private PruningMode m_pruningMode;
	Instances validationSet;
	private List<Rule> rules = new ArrayList<Rule>();

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO: implement this method
		// do some pre-processing
		// call buildTree
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}

	public void setValidation(Instances validation) {
		validationSet = validation;
	}

	@Override
	public double classifyInstance(Instance instance) {
		// TODO: implement this method
		return 0;
	}

	/**
	 * Builds the decision tree on given data set using either a recursive or
	 * queue algorithm.
	 * 
	 * @param instances
	 */
	private void buildTree(Instances instances) {

	}

	/**
	 * Calculate the average on a given instances set (could be the training,
	 * test or validation set). The average error is the total number of
	 * classification mistakes on the input instances set and divides that by
	 * the number of instances in the input set.
	 * 
	 * @param instances
	 * @return Average error
	 */
	public double calcAvgError(Instances instances) {
		double error = 0;
		double numMistake = 0;
		int numInstances = instances.numInstances();
		for (int i = 0; i < numInstances; i++) {
			Instance curInstance = instances.instance(i);
			// if real value differs from predicted value, its a mistake!
			if (curInstance.classValue() != classifyInstance(curInstance)){
				numMistake++;
			}
		}
		error = numMistake / (double) numInstances;
		return error;
	}

	/**
	 * calculates the information gain of splitting the input data according to
	 * the attribute.
	 * 
	 * @param instances
	 * @return The information gain
	 */
	private double calcInfoGain(Instances instances) {
		return 0;
	}

	/**
	 * Calculates, for set of instances, their probabilities in preparation to
	 * calculate purity (entropy), for every possible attribute
	 *
	 * @param instances
	 *            set of instances (in certain node, probably)
	 * @return probabilities of positive class (1) for each possible attribute
	 */
	private double[] calcPositiveProbabilities(Instances instances) {
		int numOfAttributes = instances.numAttributes();
		int numOfInstances = instances.numInstances();
		int classIndex = instances.classIndex();
		int numOfYes = 0;
		double[] probabilities = new double[numOfAttributes];

		// runs on all possible attributes, as long as it's not the classIndex
		// and for each attribute sums the num of "yes"s (1)
		for (int i = 0; i < numOfAttributes; i++) {
			if (i != classIndex) {
				for (int j = 0; j < numOfInstances; j++) {
					if (instances.instance(j).classValue() == 1) {
						numOfYes++;
					}
				}
				// after summing up all the "yes"s for every instance
				// for the given attribute, calculates the probability
				// and stores in the array
				probabilities[i] = numOfYes / (double) numOfInstances;
				numOfYes = 0; // zeros sum of "yes"s before next iteration
			}
		}
		return probabilities;
	}

	/**
	 * Calculates the entropy of a random variable where all the probabilities
	 * of all of the possible values it can take are given as input.
	 * 
	 * @param probabilities
	 *            - A set of probabilities
	 * @return The entropy
	 */
	private double calcEntropy(double[] probabilities) {
		int numOfInstances = probabilities.length;
		double entropy;
		double probOfCurrEvent;
		double tempCalcSi0; // these are the NOs
		double tempCalcSi1; // these are the YESs - what is calculated directly
							// in the given array

		// calculates
		for (int i = 0; i < numOfInstances; i++) {
			tempCalcSi1 = probabilities[i];
			entropy = ((tempCalcSi1 * (Math.log(tempCalcSi1) / Math.log(2.0)))
					+ (probOfNo * (Math.log(probOfNo) / Math.log(2.0))));
		}

		//// calculates prob (using log tricks to assure base 2) according to
		//// given formula
		// currAttributeProb = - ((probOfYes * (Math.log(probOfYes) /
		//// Math.log(2.0)))
		// + (probOfNo * (Math.log(probOfNo) / Math.log(2.0))));

		return 0;
	}

	/**
	 * Calculates the chi square statistic of splitting the data according to
	 * this attribute as learned in class.
	 * 
	 * @param instances
	 *            - a subset of the training data
	 * @param attributeIndex
	 * @return The chi square score
	 */
	private double calcChiSquare(Instances instances, int attributeIndex) {
		// xj is the attribute at index j (attributeIndex)
		int numValues = instances.attribute(attributeIndex).numValues();
		int numInstances = instances.numInstances();
		// number of instances for which attribute value at (j) = val(f) [Df]
		int numInstancesWithCurValue = 0;
		// Positives: number of instances for which (attVal=f) and (Y = 1) [Pj]
		int numInstanceswithFAndPos = 0;
		// Negatives: number of instances for which (attVal=f) and (Y = 0) [Nj]
		int numInstanceswithFAndNeg = 0;
		double posE, negE;
		double chiSquare = 0;

		// calculate number of positive and negative instances
		int numPositive = 0;
		int numNegative = 0;
		for (int i = 0; i < numInstances; i++) {
			Instance curInstance = instances.instance(i);
			if (curInstance.classValue() == 1) {
				numPositive++;
			} else {
				numNegative++;
			}

		}
		// TODO: this can be replaces with Adars calcProbabilities for
		// attributeIndex = classIndex
		// probability for classification (1/0)
		double Py0 = numNegative / (double) numInstances;
		double Py1 = numPositive / (double) numInstances;

		// going over all possible values
		for (int f = 0; f < numValues; f++) {
			double tempCalc = 0;
			// calculating number of instances which j attribute value is f
			for (int i = 0; i < numInstances; i++) {
				Instance curInstance = instances.instance(i);
				if (curInstance.attribute(attributeIndex).value(f) == instances.attribute(attributeIndex).value(f)) {
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

			// making sure not to divide by 0
			if ((posE != 0) && (negE != 0)) {
				tempCalc = (Math.pow((numInstanceswithFAndPos - posE), 2) / posE)
						+ (Math.pow((numInstanceswithFAndNeg - negE), 2) / negE);
			} else if ((posE != 0) && (negE == 0)) {
				// will not happen but just in case
				tempCalc = Math.pow((Math.pow((numInstanceswithFAndPos - posE), 2) / posE), 2);
			} else if ((posE == 0) && (negE != 0)) {
				// will not happen but just in case
				tempCalc = Math.pow((Math.pow((numInstanceswithFAndNeg - negE), 2) / negE), 2);
			} else {
				// happens when the number of instances where 𝑥𝑗=𝑓 [Df] is 0
				tempCalc = 0;
			}
			// adding calculation to chi square and zeros counters
			chiSquare += tempCalc;
			numInstancesWithCurValue = 0;
			numInstanceswithFAndPos = 0;
			numInstanceswithFAndNeg = 0;
		}
		return chiSquare;
	}

	/**
	 * Prunning the tree by using Chi square test in order to decide whether to
	 * prune a branch of the tree or not. We compare resulted Chi square with
	 * number from chi squared chart in the row for 8 degrees of freedom (which
	 * is the number of attributes in the cancer data minus 1) and the column
	 * for 0.95 confidence level.
	 */
	private void chiSquarePrunning() {
		double CHI_SQUARE_LIMIT = 15.51;
		// PAY ATTENTION – where you need to perform this test, what you should
		// do if the result is to prune.

	}

	/**
	 * Prunning the tree by checking if removing a rule improve the result.
	 * After complete building the tree you will go over all the rules and check
	 * if removing a rule will improve the error on the validation set. Pick the
	 * best rule to remove according to the error on the validation set and
	 * remove it from the rule set. Stop removing rules when there is no
	 * improvement.
	 * 
	 * @param validationSet
	 */
	private void rulePrunning(Instances validationSet) {
		// PAY ATTENTION – for how you loops over the rule, how you remove rules
		// during this loop, how you decide to stop.

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
