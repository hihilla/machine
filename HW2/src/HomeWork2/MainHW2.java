package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork2.DecisionTree.PruningMode;
import weka.core.Instances;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * 
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

		// builds Decision Trees each time with different pruning method
		DecisionTree treeWithNoPrunning = new DecisionTree();
		treeWithNoPrunning.setPruningMode(PruningMode.None);
		treeWithNoPrunning.setValidation(validationCancer);
		treeWithNoPrunning.buildClassifier(trainingCancer);
		
		DecisionTree treeWithChiPrunning = new DecisionTree();
		treeWithChiPrunning.setPruningMode(PruningMode.Chi);
		treeWithChiPrunning.setValidation(validationCancer);
		treeWithChiPrunning.buildClassifier(trainingCancer);
		
		DecisionTree treeWithRulePrunning = new DecisionTree();
		treeWithRulePrunning.setPruningMode(PruningMode.Rule);
		treeWithRulePrunning.setValidation(validationCancer);
		treeWithRulePrunning.buildClassifier(trainingCancer);
		
		// printing output
		// No Pruning
		System.out.println("Decision Tree with No pruning");
		System.out.println("The average train error of the decision tree is " 
							+ treeWithNoPrunning.calcAvgError(trainingCancer));
		System.out.println("The average test error of the decision tree is " 
							+ treeWithNoPrunning.calcAvgError(testingCancer));
		System.out.println("The amount of rules generated from the tree " 
							+ treeWithNoPrunning.getNumRules());
		
		// Chi Pruning
		System.out.println("Decision Tree with Chi pruning");
		System.out.println("The average train error of the decision tree is " 
							+ treeWithChiPrunning.calcAvgError(trainingCancer));
		System.out.println("The average test error of the decision tree is " 
							+ treeWithChiPrunning.calcAvgError(testingCancer));
		System.out.println("The amount of rules generated from the tree " 
							+ treeWithChiPrunning.getNumRules());
		
		//  Rule Pruning
		System.out.println("Decision Tree with Rule pruning");
		System.out.println("The average train error of the decision tree is " 
							+ treeWithRulePrunning.calcAvgError(trainingCancer));
		System.out.println("The average test error of the decision tree is " 
							+ treeWithRulePrunning.calcAvgError(testingCancer));
		System.out.println("The amount of rules generated from the tree " 
							+ treeWithRulePrunning.getNumRules());
	}
}
