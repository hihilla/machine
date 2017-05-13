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

	public BasicRule(int index, int value) {
		this.attributeIndex = index;
		this.attributeValue = value;
	}
	public BasicRule() {
		this.attributeIndex = -1;
		this.attributeValue = -1;
	}
}

class Rule {
	List<BasicRule> basicRule;
	double returnValue;

	public Rule() {
		basicRule = new ArrayList<BasicRule>();
		returnValue = -1;
	}

	public void add(BasicRule bRule) {
		basicRule.add(bRule);
	}
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex = -1;
	double returnValue;
	Rule nodeRule = new Rule();

	// Construct an empty node
	public Node() {
		this.parent = null;
		this.returnValue = -1;
		this.children = null;
		this.attributeIndex = -1;
	}

	// Construct an almost full node
	public Node(Node parent, int attributeIndex) {
		this.children = null;
		this.parent = parent;
		this.attributeIndex = attributeIndex;
		this.returnValue = -1;
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
		// build a tree - sets rootNode as its root
		rootNode = buildTree(arg0);

		// prune according to pruning mode:
		switch (m_pruningMode) {
		case Rule:
			// rule pruning:
			rulePrunning();
			break;
		case Chi:
			// chi pruning: already pruned!!!
			break;
		default:
			// no pruning!
			break;
		}
	}

	/**
	 * Builds the decision tree on given data set using a recursive algorithm
	 * 
	 * 
	 * @param instances
	 */
	private Node buildTree(Instances instances) {
		// generate empty node for root
		Node node = new Node();
		if (instances.size() == 0) {
			return node;
		}
		recBuidTree(instances, node);
		return node;
	}

	/**
	 * Recursively builds a tree with given node as root, divides given
	 * instance. Finds best attribute to split according to, using info gain.
	 * Splits given instances according to their value at bestAttribute index.
	 * Stopping when there are no instances so split, they all have the same
	 * classification (entropy 0 - pure split), or when prunning mode is chi,
	 * and we reached the chi square limit.
	 * 
	 * @param instances
	 *            - instances to split in the tree
	 * @param node
	 *            - to be root of this tree
	 */
	private void recBuidTree(Instances instances, Node node) {
		if (instances.size() == 0) {
			return;
		}

		double[] probs = calcProbabilities(instances);
		if (probs[0] == -1) {
			return;
		}
		if (probs[1] == 0 || probs[1] == 1) {
			// it is a leaf!!!
			node.returnValue = probs[1];
			node.nodeRule.returnValue = probs[1];
			rules.add(node.nodeRule);
			return;
		} else {
			// System.out.println(instances.size());
			// node isn't a leaf, setting its return value
			node.returnValue = Math.round(probs[1]);
			node.nodeRule.returnValue = Math.floor(probs[1]);
			if (calcEntropy(probs[0], probs[1]) == 0.0) {
				rules.add(node.nodeRule);
				return;
			}
		}

		// finding best attribute to split by
		int bestAttribute = findBestAttribute(instances);
		
		if (bestAttribute == -1) {
			// no way of splitting! done!
			rules.add(node.nodeRule);
			return;
		}
		// if prunning mode is chi, check to prune
		if (m_pruningMode == PruningMode.Chi) {
			double chiSquare = calcChiSquare(instances, bestAttribute);
			if (chiSquare < CHI_SQUARE_LIMIT) {
				// finish here!!
				rules.add(node.nodeRule);
				return;
			}
			System.out.println(chiSquare);
		}

		// this node should be a root of its own sub tree
		node.attributeIndex = bestAttribute;
		int numValues = instances.attribute(bestAttribute).numValues();
		Node[] children = new Node[numValues];
		node.children = children;
		Instances[] splitInstances = new Instances[numValues];
		// split instances according to their value at bestAttribute index
		for (int i = 0; i < numValues; i++) {
			splitInstances[i] = new Instances(instances, 0);
		}
		for (int i = 0; i < instances.size(); i++) {
			int index = (int) instances.instance(i).value(bestAttribute);
			splitInstances[index].add(instances.instance(i));
		}

		for (int i = 0; i < numValues; i++) {
			// create a node for this child with node as parent,
			// bestAttribute as attributeIndex, and a rule from this Attribute
			// index
			BasicRule childBasicRule = new BasicRule(bestAttribute, i);
			children[i] = new Node(node, bestAttribute);
			children[i].nodeRule.basicRule = new ArrayList<BasicRule>(node.nodeRule.basicRule);
			children[i].nodeRule.basicRule.add(childBasicRule);
			if (splitInstances[i].size() == 0) {
				children[i].returnValue = node.returnValue;
			}
			// Recursively build a subtree for this child
			recBuidTree(splitInstances[i], children[i]);
		}

	}

	/**
	 * Find max value in given array
	 * 
	 * @param arr
	 * @return the max value
	 */
	private double findMax(double[] arr) {
		double max = arr[0];
		int maxIndex = 0;
		for (int i = 1; i < arr.length; i++) {
			if (arr[i] > max) {
				max = arr[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	/**
	 * Calculate info gain for each attribute and find the attribute that gives
	 * min info gain
	 * 
	 * @param instances
	 * @param numAttributes
	 * @return
	 */
	private int findBestAttribute(Instances instances) {
		if (instances.size() == 0) {
			return -1;
		}
		int numAttributes = instances.numAttributes();
		int bestAttribute = -1;
		double goodInfoGain = -1;
		for (int i = 0; i < numAttributes; i++) {
			if (i != instances.classIndex()) {
				double tempInfoGain = calcInfoGain(instances, i);
				if (tempInfoGain > goodInfoGain) {
					goodInfoGain = tempInfoGain;
					bestAttribute = i;
				}
			}
		}

		return (goodInfoGain > 0) ? bestAttribute : -1;
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
		if (attributeIndex == -1) {
			return 0;
		}
		int numInstances = instances.numInstances();
		int numValues = instances.attribute(attributeIndex).numValues();
		// count how many instances has classification 0, and how many has 1
		double[] countClassifications = new double[2];
		// count how many instances has each possible balue at attributeIndex
		double[] countValuesOfAttribute = new double[numValues];
		// for each possible value count appearences of classifications
		double[][] countClassForValues = new double[numValues][2];
		// calculate:
		for (Instance inst : instances) {
			countClassifications[(int) inst.classValue()]++;
			countValuesOfAttribute[(int) inst.value(attributeIndex)]++;
			countClassForValues[(int) inst.value(attributeIndex)][(int) inst.classValue()]++;
		}
		double p0 = countClassifications[0] / numInstances;
		double p1 = countClassifications[1] / numInstances;
		double totalEntropy = calcEntropy(p0, p1);

		// calculate conditional entropy
		double condEntropy = 0;
		for (int i = 0; i < numValues; i++) {
			double probForVal_i = countValuesOfAttribute[i] / numInstances;
			double tempEntropy = calcEntropy(countClassForValues[i][0] / countValuesOfAttribute[i],
					countClassForValues[i][1] / countValuesOfAttribute[i]);
			if (probForVal_i == 0.0) {
				continue;
			}
			condEntropy += (probForVal_i * tempEntropy);
		}

		return totalEntropy - condEntropy;
	}

	/**
	 * Calculates the entropy of a random variable where all the probabilities
	 * of all of the possible values it can take are given as input.
	 * 
	 * @param probabilities
	 *            - A set of probabilities for a certain attribute
	 * @return entropy for this set of instances according to the attribute
	 */
	private double calcEntropy(double p0, double p1) {
		double entropy = 0;
		if (p0 != 0) {
			entropy += p0 * (Math.log(p0) / Math.log(2));
		}
		if (p1 != 0) {
			entropy += p1 * (Math.log(p1) / Math.log(2));
		}
		entropy *= (-1);
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
		int numClasses = instances.numClasses();
		// number of instances in the instances set
		int numInstances = instances.numInstances();
		double[] probabilities = new double[numClasses];

		// if there are no instances, returns meaningless array
		if (numInstances < 1) {
			for (int i = 0; i < probabilities.length; i++) {
				probabilities[i] = -1;
			}
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
			probabilities[i] = probabilities[i] / numInstances;
		}
		return probabilities;
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
//		int numInstances = instances.numInstances();
//		int numValues = instances.attribute(attributeIndex).numValues();
//		int[] classApearences = new int[2];
//		// for each possible value of attributeIndex - count each class
//		int[] classApearences0 = new int[numValues]; 
//		int[] classApearences1 = new int[numValues];
//		int[] classApearencesForValues = new int [numValues];
//		
//		for (Instance inst : instances) {
//			int attVal = (int) inst.value(attributeIndex);
//			if (inst.classValue() == 0) {
//				classApearences0[attVal]++;
//			} else {
//				classApearences1[attVal]++;
//			}
//			classApearencesForValues[attVal]++;
//			classApearences[(int) inst.classValue()]++;
//		}
//		// probability for each class
//		double p0 = classApearences[0] / (double) numInstances;
//		double p1 = classApearences[1] / (double) numInstances;
//		
//		double e0 = classApearences[0] * p0;
//		double e1 = classApearences[1] * p1;
//		double chi = 0;
//		for (int i = 0; i < numValues; i++) {
//			double tempCalc0 = 0;
//			if (e0 != 0) {
//				tempCalc0 = Math.pow((classApearences0[i] - e0), 2) / e0;
//			}
//			double tempCalc1 = 0;
//			if (e1 != 0) {
//				tempCalc1 = Math.pow((classApearences1[i] - e1), 2) / e1;
//			}
//			chi += tempCalc0 + tempCalc1;
//		}
//
//		return chi;
		
		// xj is the attribute at index j (attributeIndex)
		int numValues = instances.attribute(attributeIndex).numValues();
		int numInstances = instances.numInstances();
		// number of instances for which attribute value at (j) = val(f) [Df]
		int df = 0;
		// Positives: number of instances for which (attVal=f) and (Y = 1) [Pj]
		int pf = 0;
		// Negatives: number of instances for which (attVal=f) and (Y = 0) [Nj]
		int nf = 0;
		double E0, E1;
		double chiSquare = 0;

		// probability for classification (1/0)
		double[] probabilitiesForClass = calcProbabilities(instances);
		double Py0 = probabilitiesForClass[0];
		double Py1 = probabilitiesForClass[1];
		if (Py0 == -1) {
			return 0;
		}

		// going over all possible values for attribute at attributeIndex
		for (double f = 0; f < numValues; f++) {
			df = pf = nf = 0;
			double tempCalc = 0;
			// calculating number of instances which j attribute value is f
			// calculate df, nf, pf
			for (int i = 0; i < numInstances; i++) {
				Instance curInstance = instances.instance(i);
				if (curInstance.value(attributeIndex) == (double) f) {
					df++;
					if (curInstance.classValue() == 0) {
						pf++;
					} else {
						nf++;
					}
				}
			}
			E0 = df * Py1;
			E1 = df * Py0;

			// making sure not to divide by 0
			if (df != 0) {
				tempCalc = (Math.pow((pf - E0), 2) / E0) + (Math.pow((nf - E1), 2) / E1);
			} else {
				// happens when the number of instances where 𝑥𝑗=𝑓 [Df] is 0
				tempCalc = 0;
			}
			// adding calculation to chi square and zeros counters
			chiSquare += tempCalc;

		}
		return chiSquare;
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
	 */
	private void rulePrunning() {
		// number of rules
		int rulesNum = this.rules.size();
		// current best error
		double currBestErr = calcAvgError(validationSet);
		// to hold error after removing a single rule
		double currErr;
		// a rule pulled out to be checked if the error is better without it
		Rule extractRule;
		int counterOfPruns = 0;
		boolean rulesUpdates = true;

		while (rulesUpdates) {
			for (int i = rulesNum - 1; i >= 0; i--) {
				// removes a rule from set of rules, check the current
				// error (without the rule)
				// System.out.println("i is " + i);
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
			} else {
				// updates the size of the rules list towards next iteration
				rulesNum = rules.size();
				counterOfPruns = 0;
			}
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
		
//		int[] classApearences = new int[2];
//		int maxConsecutiveConditions = 0;
//		int bestRuleIndex = 0;
//		// iterating all Rules
//		for (int i = 0; i < numRules; i++) {
//			int numConsecutiveConditions = 0;
//			int numBasicRules = rules.get(i).basicRule.size();
//			// iterating Basic Rules
//			for (int j = 0; j < numBasicRules; j++) {
//				BasicRule curBasRule = rules.get(i).basicRule.get(j);
//				if (curBasRule.attributeValue == instance.value(curBasRule.attributeIndex)){
//					// instance applies basir rule!
//					numConsecutiveConditions++;
//				} else {
//					// instance dosn't apply basic rule!
//					if (numConsecutiveConditions > maxConsecutiveConditions) {
//						// new best rule
//						classApearences[0] = classApearences[1] = 0;
//						maxConsecutiveConditions = numConsecutiveConditions;
//						bestRuleIndex = i;
//						classApearences[(int) rules.get(i).returnValue]++;
//					} else if (numConsecutiveConditions == maxConsecutiveConditions) {
//						classApearences[(int) rules.get(i).returnValue]++;
//					}
//					break;
//				}
//			}
//			// sanity check:
//			if (numConsecutiveConditions == numBasicRules) {
//				// instance applies entire rule. will not get here, but better
//				// safe than sorry
//				return rules.get(bestRuleIndex).returnValue;
//			}
//		}
//		
//		
//	 	return (classApearences[0] > classApearences[1]) ? 0 : 1;
	 	
//		 if more then one rule has the same number of consecutive met
//		 conditions
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
		 if (curBasicRule.attributeValue !=
		 instance.value(curBasicRule.attributeIndex)) {
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
		 } else if (curConsecutiveCondition ==
		 largestNumOfConsecutiveConditions) {
		 // same number of consecutive conditions as a previous Rule
		 // adding this Rule to list of Rules
		 moreThenOneRule = true;
		 suitableRules.add(curRule);
		 if (suitableRules.indexOf(mostSuitableRule) == -1) {
		 suitableRules.add(mostSuitableRule);
		 }
		 }
		 }
		 if (moreThenOneRule) {
		 double[] retValues = new
		 double[instance.classAttribute().numValues()];
		 int numOfSuitableRules = suitableRules.size();
		 // System.out.println(numOfSuitableRules);
		 // map return values of all rules
		 for (int i = 0; i < numOfSuitableRules; i++) {
		 double value = suitableRules.get(i).returnValue;
		 retValues[(int) value]++;
		 }
		
		 // return the most common return value
		 return findMax(retValues);
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
		int numInstances = instances.numInstances();
		double numMistake = 0;
		for (Instance instance : instances) {
			if (instance.classValue() != classifyInstance(instance)) {
				numMistake++;
			}
		}
		return numMistake / numInstances;
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

	/**
	 * @return the number of rules in decision tree
	 */
	public int getNumRules() {
		return rules.size();
	}

	public void printTree() {
		
		recPrintTree(rootNode, 0);
//		 System.out.print("nodeID:" + node.id + " ");
//		 if (node.parent != null){
//		 System.out.print("parent ID:" + node.parent.id + " ");
//		 }
//		System.out.print("Attribute " + node.attributeIndex + " ");
//		System.out.print("returnVal " + node.returnValue + " ");
//		// System.out.print("numInst " + node.nodesInstances.numInstances() + "
//		// ");
//		if (node.children == null || node.children.length == 0) {
//			System.out.println("  I'm leaf   ");
//			return;
//		}
//
//		System.out.println();
//		for (int i = 0; i < node.children.length; i++) {
//			if (node.children[i] == null)
//				continue;
//			printTree(node.children[i]);
//		}
//		return;

	}

	private void recPrintTree(Node node, int level) {
		for (int i = 0; i < level; i++) {
			System.out.print(" - ");
		}
		System.out.println("> " + node.attributeIndex + " value: " + node.returnValue);
		if (node.returnValue != 1.0 && node.returnValue != 0) {
			System.out.println("HA?!");
		}
		if (node.children != null) {
			for (Node child : node.children) {
				recPrintTree(child, level + 1);
			}
		}
	}
	
	public void printRules(){
		for (Rule rule : rules) {
			System.out.println();
			for (BasicRule bRule : rule.basicRule) {
				System.out.print(bRule.attributeIndex + " : " + bRule.attributeValue + " -> ");
			}
			System.out.print(rule.returnValue);
		}
	}
}
