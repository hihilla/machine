package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import com.sun.javafx.css.CalculatedValue;
import com.sun.org.glassfish.gmbal.Description;

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
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
		//builds Decision Trees each time with different pruning method
		DecisionTree treeWithNoPrunning = new DecisionTree();
		treeWithNoPrunning.setPruningMode(PruningMode.None);
		treeWithNoPrunning.setValidation(validationCancer);
		treeWithNoPrunning.buildClassifier(trainingCancer);
		System.out.println("no prunning avg error");
		System.out.println(treeWithNoPrunning.calcAvgError(testingCancer));
		
		DecisionTree treeWithChiPruning = new DecisionTree();
		treeWithChiPruning.setPruningMode(PruningMode.Chi);
		treeWithChiPruning.setValidation(validationCancer);
		treeWithChiPruning.buildClassifier(trainingCancer);
		System.out.println("chi prunning avg error");
		System.out.println(treeWithChiPruning.calcAvgError(testingCancer));
		
//		DecisionTree treeWithRulePruning = new DecisionTree();
//		treeWithRulePruning.setPruningMode(PruningMode.Rule);
//		treeWithRulePruning.setValidation(validationCancer);
//		treeWithRulePruning.buildClassifier(trainingCancer);
//		
		double aveTrainErr = treeWithNoPrunning.calcAvgError(trainingCancer);
		double aveTestErr = treeWithNoPrunning.calcAvgError(testingCancer);
		int numOfRules = treeWithNoPrunning.getNumRules();
//		System.out.println("Decision Tree with No prunning");
//		System.out.println("The average train error of the decision tree is "
//							+ aveTrainErr);
//		System.out.println("The average test error of the decision tree is "
//							+ aveTestErr);
//		System.out.println("The amount of rules generated from the tree " 
//							+ numOfRules);
//		
		aveTrainErr = treeWithChiPruning.calcAvgError(trainingCancer);
		aveTestErr = treeWithChiPruning.calcAvgError(testingCancer);
		numOfRules = treeWithChiPruning.getNumRules();
		System.out.println("Decision Tree with Chi prunning");
		System.out.println("The average train error of the decision tree "
							+ "with Chi pruning is "
							+ aveTrainErr);
		System.out.println("The average test error of the decision tree" 
							+ "with Chi pruning is "	
							+ aveTestErr);
		System.out.println("The amount of rules generated from the tree " 
				+ numOfRules);
//		
//		aveTrainErr = treeWithRulePruning.calcAvgError(trainingCancer);
//		aveTestErr = treeWithRulePruning.calcAvgError(testingCancer);
//		numOfRules = treeWithRulePruning.getNumRules();
//		System.out.println("Decision Tree with Rule prunning");
//		System.out.println("The average train error of the decision tree "
//							+ "with Rule pruning is "
//							+ aveTrainErr);
//		System.out.println("The average test error of the decision tree"
//							+ "with Rue pruning"
//							+ aveTestErr);
//		System.out.println("The amount of rules generated from the tree " 
//				+ numOfRules);
//		
		
	}
}
