package homework2;

import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

// Results on cancer dataset:
// --------------------------
// Without pruning:
// Average error on testingData = 0.280000
// Average error on trainingData = 0.029787
// With pruning:
// Average error on testingData = 0.240000
// Average error on trainingData = 0.200000

@SuppressWarnings("serial")
public class DecisionTree extends Classifier{
	
	private class Node {
		public int attributeIndex;		// On what attribute did we split this node?
		public Node[] children; 		// The different branches for the attribute
		public boolean isLeaf; 			// Did we reach classification?
		public double classValue; 		// What is the classification?
		
		// Constructor for leaf nodes
		public Node(double classValue) {
			this.attributeIndex = -1;
			this.children = null;
			this.isLeaf = true;
			this.classValue = classValue;
		}
		
		// Constructor for internal nodes
		public Node(int attributeIndex, Node[] children) {
			this.attributeIndex = attributeIndex;
			this.children = children;
			this.isLeaf = false;
			this.classValue = -1;
		}
		
	}
	
	private Node tree; 									// The decision tree
	private boolean m_pruningMode = false;				// To prune or not to prune
	
	private final double CHI_SQUARE_THRESHOLD = 2.733; 	// For cancer dataset

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		tree = buildTree(arg0);
	}
	
	/**
	 * Build the decision tree from the dataset.
	 * 
	 * @param trainingData
	 * @return
	 */
	private Node buildTree(Instances trainingData) {
		
		// 1st stopping condition - check if they all have the same class value
		boolean sameClassValue = true;
		double classValue = trainingData.instance(0).classValue();
		for (int i=1; i<trainingData.numInstances(); i++) {
			if (trainingData.instance(i).classValue() != classValue) {
				sameClassValue = false;
				break;
			}
		}
		if (sameClassValue) {
			return new Node(classValue);
		}
		
		// 2nd stopping condition - check if they all have the same attributes
		boolean sameAttributes = true;
		for (int i=0; i<trainingData.numAttributes() - 1; i++) {
			double attributeValue = trainingData.instance(0).value(i);
			for (int j=1; j<trainingData.numInstances(); j++) {
				if (trainingData.instance(j).value(i) != attributeValue) {
					sameAttributes = false;
					break;
				}
			}
		}
		if (sameAttributes) {
			double[] arr = trainingData.attributeToDoubleArray(trainingData.classIndex());
			return new Node(findMajority(arr));
		}
		
		// Choose the best attribute to split on
		int bestAtt = findBestAttribute(trainingData);

		// Perform pre-pruning?
		if (m_pruningMode) {
			double chi = calcChiSquare(trainingData, bestAtt);
			if (chi < CHI_SQUARE_THRESHOLD) {
				double[] arr = trainingData.attributeToDoubleArray(trainingData.classIndex());
				return new Node(findMajority(arr));
			}
		}
		
		// Split on best attribute
		Instances[] split = new Instances[trainingData.attribute(bestAtt).numValues()];
		for (int i=0; i<split.length; i++) {
			split[i] = new Instances(trainingData);
			for (int j=trainingData.numInstances()-1; j>=0; j--) {
				if ((int) split[i].instance(j).value(bestAtt) != i) {
					split[i].delete(j);
				}
			}
		}
		
		// Build tree recursively
		Node[] children = new Node[split.length];
		for (int i=0; i<split.length; i++) {
			if (split[i].numInstances() != 0) {
				children[i] = buildTree(split[i]);
			} else {
				double[] arr = trainingData.attributeToDoubleArray(trainingData.classIndex());
				double val = findMajority(arr);
				children[i] = new Node(val);
			}
		}
		
		return new Node(bestAtt, children);
		
	}
	
	/**
	 * Find the best attribute to split on (with the highest info gain) in the dataset.
	 * 
	 * @param trainingData
	 * @return
	 */
	private int findBestAttribute(Instances dataset) {
		int bestAttIndex = 0;
		double bestAttIG = calcInfoGain(dataset, 0);
		for (int i=1; i<dataset.numAttributes()-1; i++) {
			double ig = calcInfoGain(dataset, i);
			if (ig > bestAttIG) {
				bestAttIG = ig;
				bestAttIndex = i;
			}
		}
		return bestAttIndex;
	}
	
	/**
	 * Calculate the entropy of a chosen attribute.
	 * 
	 * @param probabilities - the set of probabilities for all possible values of the chosen attribute
	 * @return the entropy of that attribute
	 */
	private double calcEntropy(double[] probabilities) {
		double sum = 0;
		double log2 = Math.log(2);
		for (int i=0; i<probabilities.length; i++) {
			double p = probabilities[i];
			if (p != 0) {
				sum += (-1) * p * Math.log(p) / log2;
			}
		}
		return sum;
	}
	
	/**
	 * Calculate the probabilities for all possible values of attribute #attributeIndex in the dataset.
	 * 
	 * @param dataset
	 * @param attributeIndex
	 * @return
	 */
	private double[] calcProbabilities(Instances dataset, int attributeIndex) {
		
		// Create a new array the size of all possible values of the chosen attribute
		double[] probabilities = new double[dataset.attribute(attributeIndex).numValues()];
		
		// Check for empty dataset
		if (dataset.numInstances() == 0) {
			return probabilities;
		}
		
		// Create a histogram of the possible values
		for (int i=0; i<dataset.numInstances(); i++) {
			probabilities[(int) dataset.instance(i).value(attributeIndex)]++;
		}
		
		// Calculate the probabilities from the histogram
		for (int i=0; i<probabilities.length; i++) {
			probabilities[i] /= dataset.numInstances();
		}
		
		return probabilities;
		
	}
	
	/**
	 * Calculate the probabilities for all possible values of attribute #attributeIndex in the dataset,
	 * but only on instances where attribute #filterAttributeIndex has value filterAttributeValue.
	 * 
	 * @param dataset
	 * @param attributeIndex
	 * @param attributeValue
	 * @return
	 */
	private double[] calcProbabilitiesWithFilter(Instances dataset, int attributeIndex, int filterAttributeIndex, double filterAttributeValue) {
		
		// Create a new copy of the dataset
		Instances newDataset = new Instances(dataset);
		
		// But keep only the instances where attribute #attributeIndex == attributeValue
		for (int i=newDataset.numInstances()-1; i>=0; i--) {
			if (newDataset.instance(i).value(filterAttributeIndex) != filterAttributeValue) {
				newDataset.delete(i);
			}
		}
		
		// Now calculate the probabilities of attribute #attributeIndex on the new dataset
		double[] probabilities = calcProbabilities(newDataset, attributeIndex); 
		
		return probabilities;
		
	}
	
	/**
	 * Calculate the information gain of attribute #attributeIndex in the dataset.
	 * 
	 * @param dataset
	 * @param attributeIndex
	 * @return
	 */
	private double calcInfoGain(Instances dataset, int attributeIndex) {
		double[] probabilities = calcProbabilities(dataset, attributeIndex);
		double sum = 0;
		for (int i=0; i<dataset.attribute(attributeIndex).numValues(); i++) {
			double[] probabilities2 = calcProbabilitiesWithFilter(dataset, dataset.classIndex(), attributeIndex, i);
			sum += probabilities[i] * calcEntropy(probabilities2);
		}
		return calcEntropy(probabilities) - sum;
	}
	
	/**
	 * Classify an instance.
	 * 
	 * @param x
	 * @return
	 */
	public double classify(Instance x) {
		Node node = tree;
		while (node.isLeaf == false) {
			node = node.children[(int) x.value(node.attributeIndex)];
		}
		return node.classValue;
	}
	
	/**
	 * Calculate the average error of the classifier on the test set.
	 * 
	 * @param testset
	 * @return
	 */
	public double calcAvgError(Instances testset) {
		double numMistakes = 0;
		for (int i=0; i<testset.numInstances(); i++) {
			double myClassValue = classify(testset.instance(i));
			double realClassValue = testset.instance(i).classValue();
			if (myClassValue != realClassValue) {
				numMistakes++;
			}
		}
		return numMistakes / testset.numInstances();
	}
	
	/**
	 * Find the majority value of an array.
	 * 
	 * @param a
	 * @return
	 */
	private double findMajority(double[] a) {
		
	    if (a == null || a.length == 0) {
	        return 0;
	    }

	    Arrays.sort(a);

	    double previous = a[0];
	    double popular = a[0];
	    int count = 1;
	    int maxCount = 1;

	    for (int i=1; i<a.length; i++) {
	        if (a[i] == previous) {
	            count++;
	        } else {
	            if (count > maxCount) {
	                popular = a[i-1];
	                maxCount = count;
	            }
	            previous = a[i];
	            count = 1;
	        }
	    }
	    
	    return (count > maxCount) ? a[a.length-1] : popular;
	    
	}
	
	/**
	 * Turn pruning on/off.
	 * 
	 * @param pruningMode
	 */
	public void setPruningMode(boolean pruningMode){
		m_pruningMode = pruningMode;
	}
	
	/**
	 * Calculate the Chi Square statistic of attribute #attributeIndex in the dataset.
	 * 
	 * @param dataset
	 * @param attributeIndex
	 * @return
	 */
	private double calcChiSquare(Instances dataset, int attributeIndex) {
		
		double sum = 0;
		double[] y_probabilities = calcProbabilities(dataset, dataset.classIndex());
		
		// Iterate over all possible values of attribute #attributeIndex
		for (double f=0; f<dataset.attribute(attributeIndex).numValues(); f++) {
			
			int df = 0;
			int pf = 0;
			int nf = 0;
			
			// Iterate over all instances and start counting
			for (int j=0; j<dataset.numInstances(); j++) {
				if (dataset.instance(j).value(attributeIndex) == f) {
					df++;
					if ((int) dataset.instance(j).classValue() == 0) {
						pf++;
					} else {
						nf++;
					}
				}
			}
			
			// Make sure you don't divide by 0
			if (df == 0) {
				continue;
			}
			
			double e0 = df * y_probabilities[0];
			double e1 = df * y_probabilities[1];
			
			// Sum chi square
			sum += (e0-pf)*(e0-pf)/e0 + (e1-nf)*(e1-nf)/e1; 
			
		}
		
		return sum;
	}

}
