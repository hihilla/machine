package HomeWork2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class BasicRule {
	int attributeIndex;
	int attributeValue;

	public BasicRule(int index, int value) {
		this.attributeIndex = index;
		this.attributeValue = value;
	}
}

class Rule {
	List<BasicRule> basicRule;
	double returnValue;

	public void add(BasicRule bRule) {
		basicRule.add(bRule);
	}
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	Rule nodeRule = new Rule();
	boolean marked = false;

	// Construct a general child-less node
	public Node(Node[] children) {
		this.parent = null;
		this.children = children;
		this.attributeIndex = -1;
		this.returnValue = -1;
	}

	// Construct a leaf node
	public Node(int returnValue) {
		this.parent = null;
		this.returnValue = returnValue;
		this.children = null;
		this.attributeIndex = -1;
	}
}

public class DecisionTree implements Classifier {
	private Node rootNode;

	public enum PruningMode {
		None, Chi, Rule
	};

	private PruningMode m_pruningMode;
	Instances validationSet;
	private List<Rule> rules = new ArrayList<Rule>();

	private final double CHI_SQUARE_LIMIT = 15.51;

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// build a tree
		rootNode = buildTree(arg0);
		// go over it and find each Rules returnValue
		int numOfRules = this.rules.size();
		for (int i = 0; i < numOfRules; i++) {
			rules.get(i).returnValue = findReturnValue(arg0);
		}

		// do some prunning
		// set the tree to this classObject....

	}

	/**
	 * Builds the decision tree on given data set using a recursive
	 * algorithm
	 * 
	 * 
	 * @param instances
	 */
	private Node buildTree(Instances instances) {
		int numAttributes = instances.numAttributes();

		if (sameAttributeValue(instances) || sameClassValue(instances)) {
			// all instances are getting same classification, this node is a
			// leaf.
			// find the returnValue for this leaf:
			int returnValue = findReturnValue(instances);
			return new Node(returnValue);
		}

		// getting best attribute
		int bestAttribute = findBestAttribute(instances, numAttributes);

		// create children for the node
		int numOfChildren = instances.attribute(bestAttribute).numValues();
		Node[] childs = new Node[numOfChildren];

		// define node with bestAttribute as attributeIndex and give it the
		// children
		Node node = new Node(childs);

		// divide instances to children
		Instances[] divideInstances = new Instances[numOfChildren];
		for (int i = 0; i < numOfChildren; i++) {
			divideInstances[i] = generateSubsetInstances(instances, bestAttribute, i);
		}

		// now actually create the children and their tree
		for (int i = 0; i < numOfChildren; i++) {
			if (divideInstances[i].numInstances() != 0) {
				// building a tree for a child that has instances
				childs[i] = buildTree(divideInstances[i]);
			} else {
				// find the returnValue for this leaf:
				int returnValue = findReturnValue(instances);
				// set return value and parent for this leaf
				childs[i] = new Node(returnValue);
			}
			childs[i].parent = node;
			childs[i].attributeIndex = bestAttribute;
			BasicRule childRule = new BasicRule(bestAttribute, i);
			childs[i].nodeRule.add(childRule);
		}
		return node;
	}

	private void findAllRules() {
		Rule curRule;
		List<Rule> foundRules = new ArrayList<Rule>();
		Node curNode = this.rootNode;
		
	}
	
	/**
	 * Checks if all nodes in nodes array are marked.
	 * @param nodes
	 * @return true if all are marked, if one isn't returns false
	 */
	private boolean areMarked(Node[] nodes) {
		for (Node node : nodes) {
			if (!node.marked) return false;
		}
		
		return true;
	}

	/**
	 * Iterate over the tree using a recursive function and returning all leafs
	 * 
	 * @return all leafs in the decision tree
	 */
	private Node[] findAllLeafs() {
		List<Node> lst = recFindAllLeafs(this.rootNode);
		return (Node[]) lst.toArray();
	}

	private List<Node> recFindAllLeafs(Node node) {
		List<Node> lst = new ArrayList<Node>();
		if (node.children.length == 0) {
			lst.add(node);
			return lst;
		}
		for (int i = 0; i < node.children.length; i++) {
			lst.addAll(recFindAllLeafs(node.children[i]));
		}
		return lst;
	}

	/**
	 * Find the return value according to given subset of instances
	 * 
	 * @param instances
	 * @param classIndex
	 * @param numOfClassifications
	 * @return
	 */
	private int findReturnValue(Instances instances) {
		int classIndex = instances.classIndex();
		int numOfClassifications = instances.numClasses();
		int returnValue;
		// creating an array of size (number of instances), 
		// each cell i states the
		// classification of instance i
		double[] instancesClassifications = instances.attributeToDoubleArray(classIndex);
		if (instancesClassifications == null || instancesClassifications.length == 0) {
			returnValue = 0;
		} else {
			// counting number of appearances for each classification and
			// finding the
			// classification that appears the most (max number of appearances)
			returnValue = findMax(buildHistogram(instancesClassifications, numOfClassifications));
		}
		return returnValue;
	}

	/**
	 * Checking if all given instances has the same attribute values for each
	 * attribute.
	 * 
	 * @param instances
	 * @return true if all instances has the same attribute values.
	 */
	private boolean sameAttributeValue(Instances instances) {
		int numInstances = instances.numInstances();
		int numAttribute = instances.numAttributes();

		// going over all attributes and checking for same attribute values:
		for (int i = 0; i < numAttribute; i++) {
			Instance curInstance = instances.firstInstance();
			double attributeValue = curInstance.value(0);
			for (int j = 1; j < numInstances; j++) {
				curInstance = instances.instance(i);
				if (curInstance.value(j) != attributeValue) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * Checking if all given instances has the same classification.
	 * 
	 * @param instances
	 * @return true if all instances has the same classification.
	 */
	private boolean sameClassValue(Instances instances) {
		int numInstances = instances.numInstances();

		// if all instances have the same classification:
		Instance curInstance = instances.firstInstance();
		double classValue = curInstance.classValue();
		for (int i = 1; i < numInstances; i++) {
			curInstance = instances.instance(i);
			if (curInstance.classValue() != classValue) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Create a histogram from a given array with values in range [0 - size]
	 * 
	 * @param arr
	 * @param size
	 * @return histogram
	 */
	private int[] buildHistogram(double[] arr, int size) {
		int[] histogram = new int[size];
		for (int i = 0; i < histogram.length; i++) {
			histogram[i] = 0;
		}
		for (int i = 0; i < arr.length; i++) {
			histogram[(int) arr[i]]++;
		}
		return histogram;
	}

	/**
	 * Find max value in given array
	 * 
	 * @param arr
	 * @return the max value
	 */
	private int findMax(int[] arr) {
		int max = arr[0];
		for (int i = 1; i < arr.length; i++) {
			if (arr[i] > max) {
				max = arr[i];
			}
		}
		return max;
	}

	/**
	 * Calculate info gain for each attribute and find the attribute that gives
	 * min info gain
	 * 
	 * @param instances
	 * @param numAttributes
	 * @return
	 */
	private int findBestAttribute(Instances instances, int numAttributes) {
		int bestAttribute = 0;
		double goodInfoGain = calcInfoGain(instances, 0);
		for (int i = 1; i < numAttributes; i++) {
			double tempInfoGain = calcInfoGain(instances, i);
			if (tempInfoGain < goodInfoGain) {
				goodInfoGain = tempInfoGain;
				bestAttribute = i;
			}
		}
		return bestAttribute;
	}

	/**
	 * calculates the information gain of splitting the input data according to
	 * the attribute
	 * 
	 * @param instances
	 *            subset of instances to calc according to them the infoGain
	 * @param attributeIndex
	 *            the attribute to calc for the infoGain
	 * @return
	 */
	private double calcInfoGain(Instances instances, int attributeIndex) {
		// total of all iterations of sigma
		double Sigma = 0;
		// entropy of all of the instances (first part of formula)
		double entropyS = calcEntropy(calcProbabilities(instances));
		// the array of probabilities, to be used while calculate tempSigma
		double[] probs = calcProbabilities(instances);
		//
		double subsetEntropy;

		for (int i = 0; i < probs.length; i++) {
			// hold only instances that hold the value i of the given attribute
			Instances subsetInstances = generateSubsetInstances(instances, attributeIndex, i);
			// calculates the entropy of the instances with value i
			subsetEntropy = calcEntropy(calcProbabilities(subsetInstances));
			// total inner sigma-the entropy of instances with value i *
			// probability of this value given its attribute
			Sigma += subsetEntropy * probs[i];
		}
		return (entropyS - Sigma);
	}

	/**
	 * Calculates the entropy of a random variable where all the probabilities
	 * of all of the possible values it can take are given as input.
	 * 
	 * @param probabilities
	 *            - A set of probabilities for a certain attribute
	 * @return entropy for this set of instances according to the attribute
	 */
	private double calcEntropy(double[] probabilities) {
		// hold the current probability of a certain value
		double tempValProb;
		int numOfValues = probabilities.length;
		double entropy = 0;

		// calculates for every cell of the array its part of
		// the entropy (the Sigma itself), and sums to entropy
		for (int i = 0; i < numOfValues; i++) {
			if (probabilities[i] != 0) {
				tempValProb = probabilities[i];
				entropy += (-1) * ((tempValProb * (Math.log(tempValProb) / Math.log(2.0))));
			}
		}

		return entropy;
	}

	/**
	 * Calculates, for set of instances, their probabilities for all of possible
	 * values according to a given attribute
	 * 
	 * @param instances
	 *            set of instances
	 * @param attributeIndex
	 *            attribute to check probs according to its possible values
	 * @return array of double with all possible probabilities
	 */

	private double[] calcProbabilities(Instances instances) {
		// number of possible classifications
		int numValues = instances.numClasses();
		// number of instances in the instances set
		int numInstances = instances.numInstances();
		double[] probabilities = new double[numValues];

		// if there are no instances, returns meaningless array
		if (numInstances < 1) {
			return probabilities;
		}

		// goes through all instances and gets for each the value
		// of the attribute, stores the info in the cell of the array
		// that corresponds to that possible value
		for (int i = 0; i < numInstances; i++) {
			probabilities[(int) instances.instance(i).classValue()]++;
		}

		// puts the actual probabilities in the array be dividing each
		// cell of the array by the number of possible values
		for (int i = 0; i < probabilities.length; i++) {
			probabilities[i] = probabilities[i] / numValues;
		}
		return probabilities;
	}

	/**
	 * Generates a subset of instances for which the attribute value at
	 * attributeIndex is attributeValue. Meaning for every instance in subset
	 * the value at the given attribute is equal to the given value.
	 * 
	 * @param instances
	 * @param attributeIndex
	 * @param attributeValue
	 * @return subset of instances
	 */
	private Instances generateSubsetInstances(Instances instances, int attributeIndex, double attributeValue) {
		Instances subInstances = new Instances(instances);
		int numInstances = instances.numInstances();
		// removing instances with different value
		for (int i = numInstances - 1; i <= 0; i--) {
			Instance curInstance = subInstances.instance(i);
			double curValue = curInstance.value(attributeIndex);
			if (curValue != attributeValue) {
				// value is not a match, removing from subset
				subInstances.delete(i);
			}
		}
		return subInstances;
	}

	/**
	 * Calculates the chi square statistic of splitting the data according to
	 * this attribute as learned in class.
	 * 
	 * @param instances
	 *            - a subset of the training data
	 * @param attributeIndex
	 *            - should be class index
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

		// probability for classification (1/0)
		double[] probabilitiesForClass = calcProbabilities(instances);
		double Py0 = probabilitiesForClass[0];
		double Py1 = probabilitiesForClass[1];

		// going over all possible values for attribute at attributeIndex
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
	 * PAY ATTENTION – where you need to perform this test, what you should
	 * do if the result is to prune.
	 */
	private void chiSquarePrunning() {
		Node[] leafs = findAllLeafs();
		int numOfLeafs = leafs.length;
		double chiSquare = Double.MIN_VALUE;
		// continue pruning while chiSquare is not 95% confidence
		while (chiSquare < CHI_SQUARE_LIMIT) {
			Node bestLeaf = leafs[0];
			double bestChi = Double.MIN_VALUE;
			// iterating over leafs. take out the leaf with largest chi square.
			for (int i = 0; i < numOfLeafs; i++) {
				DecisionTree tempTree = new DecisionTree();
				// simulate the prune of this leaf and check for chi square
				tempTree.buildTree(validationSet);
				tempTree.removeNode(leafs[i]);
				double curChi = tempTree.calcChiSquare(validationSet, i);
				if (curChi > bestChi) {
					// prune this leaf!!!
					bestLeaf = leafs[i];
					bestChi = curChi;
				}
			}
			// taking out leaf with best chi square
			this.removeNode(bestLeaf);
		}
		// now chi square is at 95% confidence with 8 degrees of freedom
		// now re-finding rules and return values
		// go over tree and find each Rules returnValue
		int numOfRules = this.rules.size();
		for (int i = 0; i < numOfRules; i++) {
			rules.get(i).returnValue = findReturnValue(validationSet);
		}
	}

	private void removeNode(Node node) {
		Node parent = node.parent;
		Node[] siblings = parent.children;
		Node[] childs = node.children;
		Node[] newChildren = new Node[siblings.length + childs.length - 1];
		for (int i = 0; i < childs.length; i++) {
			newChildren[i] = childs[i];
		}
		int j = 0;
		for (int i = childs.length; i < newChildren.length; i++, j++) {
			if (siblings[j] != node) {
				newChildren[i] = childs[j];
			}
		}
		parent.children = newChildren;
		node.parent = null;
	}

	/**
	 * Pruning the tree by checking if removing a rule improve the result. After
	 * complete building the tree you will go over all the rules and check if
	 * removing a rule will improve the error on the validation set. Pick the
	 * best rule to remove according to the error on the validation set and
	 * remove it from the rule set. Stop removing rules when there is no
	 * improvement. PAY ATTENTION – for how you loops over the rule, how you
	 * remove rules during this loop, how you decide to stop.
	 * 
	 * @param validationSet
	 */
	private void rulePrunning(Instances validationSet) {
		// number of rules
		int rulesNum = rules.size();
		// current best error
		double currBestErr = calcAvgError(validationSet);
		// to hold error after removing a single rule
		double currErr;
		// a rule pulled out to be checked if the error is better without it
		Rule extractRule;
		int counterOfPruns = 0;
		boolean rulesUpdates = true;

		while (rulesUpdates) {
			for (int i = rulesNum; i >= 0; i--) {
				// removes a rule from set of rules, check the current
				// error (without the rule)
				extractRule = rules.remove(i);
				currErr = calcAvgError(validationSet);
				// if there was an improvement:
				// updates the current best error to be the current error
				// adds 1 to counter
				if (currErr < currBestErr) {
					currBestErr = currErr;
					counterOfPruns++;
					// there's no improvement, returns the current rule to rules
				} else {
					rules.add(i, extractRule);
				}
			}
			// if there are no updates, this is the best set of
			// rules possible and wer're done
			// (else, gets into another iteration of pruning)
			if (counterOfPruns == 0) {
				rulesUpdates = false;
			}
			// updates the size of the rules list towards next iteration
			rulesNum = rules.size();
		}
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}

	public void setValidation(Instances validation) {
		validationSet = validation;
	}

	@Override
	/**
	 * checks the classification of a single instance
	 *
	 * @param instance
	 *            for which function checks classification
	 * @return instance's classification
	 */
	public double classifyInstance(Instance instance) {
		int numRules = this.rules.size();
		int numBasicRules;

		// Trying to find a Rule that applies.
		// If not applying a basic rule in the current Rule,
		// stop with this Rule and continue to next Rule.
		for (int i = 0; i < numRules; i++) {
			Rule curRule = this.rules.get(i);
			boolean applyRule = true;
			numBasicRules = curRule.basicRule.size();
			for (int j = 0; j < numBasicRules && applyRule; j++) {
				BasicRule curBasicRule = curRule.basicRule.get(j);
				if (curBasicRule.attributeValue != instance.value(curBasicRule.attributeIndex)) {
					applyRule = false; // Stop and continue to next Rule.
				}
			}
			if (applyRule) {
				// if after checking all basic rules in the Rule applyRule is
				// true,
				// the instance applies the Rule, return appropriate return
				// Value.
				return curRule.returnValue;
			}
		}

		// instance does not purely applies any Rule.
		// need to fine a best matched rule.
		// if more then one rule has the same number of consecutive met
		// conditions
		// insert it to the suitableRules array and choose from there.
		Rule mostSuitableRule = this.rules.get(0);
		List<Rule> suitableRules = new ArrayList<Rule>();
		int largestNumOfConsecutiveConditions = 0;
		boolean moreThenOneRule = false;
		for (int i = 0; i < numRules; i++) {
			boolean applyRule = true;
			Rule curRule = this.rules.get(i);
			int curConsecutiveCondition = 0;
			numBasicRules = curRule.basicRule.size();
			for (int j = 0; j < numBasicRules && applyRule; j++) {
				BasicRule curBasicRule = curRule.basicRule.get(j);
				if (curBasicRule.attributeValue != instance.value(curBasicRule.attributeIndex)) {
					applyRule = false; // Stop and continue to next Rule.
				} else {
					// count this Rules consecutive conditions
					curConsecutiveCondition++;
				}
			}
			if (curConsecutiveCondition > largestNumOfConsecutiveConditions) {
				// this Rule is better then last one!
				// saving this Rule and deleting previous data
				moreThenOneRule = false;
				largestNumOfConsecutiveConditions = curConsecutiveCondition;
				mostSuitableRule = curRule;
				suitableRules = new ArrayList<Rule>();
			} else if (curConsecutiveCondition == largestNumOfConsecutiveConditions) {
				// same number of consecutive conditions as a previous Rule
				// adding this Rule to list of Rules
				moreThenOneRule = true;
				suitableRules.add(curRule);
			}
		}
		if (moreThenOneRule) {
			// there are more than one rule with the largest number from the
			// previous step
			// classify with the majority of the returning values of those rules
			int numOfRules = suitableRules.size();
			int minNumOfRules = Integer.MAX_VALUE;
			mostSuitableRule = suitableRules.get(0);
			int numOfRulesToEnd = 0;
			for (int i = 0; i < numOfRules; i++) {
				Rule curRule = suitableRules.get(i);
				numOfRulesToEnd = 0;
				// iterating over Rule until meeting an unfulfilled basic rule
				boolean meetRule = true;
				BasicRule curBasicRule = curRule.basicRule.get(0);
				for (int j = 0; j < curRule.basicRule.size() && meetRule; j++) {
					curBasicRule = curRule.basicRule.get(i);
					if (curBasicRule.attributeValue != instance.value(curBasicRule.attributeIndex)) {
						// an unfulfilled basic rule met!
						meetRule = false;
					}
				}
				// curBasicRule is the unfulfilled basic rule
				// count number of basic rules until the end of this Rule
				for (int j = curRule.basicRule.indexOf(curBasicRule); j < curRule.basicRule.size(); j++) {
					numOfRulesToEnd++;
				}
				if (numOfRulesToEnd < minNumOfRules) {
					// if this Rule is closer to the end (to a leaf) then the
					// previous
					// best rule, keep it (to return at the end)
					numOfRulesToEnd = minNumOfRules;
					mostSuitableRule = curRule;
				}
			}
			// mostSuitableRule is the best rule - has the shortest path to a
			// leaf
			// among other rules that has the same number of consecutive met
			// conditions
			return mostSuitableRule.returnValue;
		} else {
			// returning the rule that meets the largest number of consecutive
			// conditions
			return mostSuitableRule.returnValue;
		}
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
			if (curInstance.classValue() != classifyInstance(curInstance)) {
				numMistake++;
			}
		}
		error = numMistake / (double) numInstances;
		return error;
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
