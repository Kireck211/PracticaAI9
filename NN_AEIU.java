import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NN_AEIU {

	static int trainingSetSize   = 0;
	static int inputLayerLength  = 0;
	static int hiddenLayerLength = 0;
	static int outputLayerLength = 0;
	static byte[][]   trainingSet         = null;
	static double[][] expectedOutputs     = null;
	static double[][] inputHiddenWeights  = null;
	static double[]   hiddenLayerValues   = null;
	static double[][] hiddenOutputWeights = null;
	static double[]    outputLayerValues  = null;
	
	static final double LEARNING_RATE = 0.8;
	static final double TEST_THRESHOLD = 1.0 / 6;
	static final double TRAIN_THRESHOLD = TEST_THRESHOLD / 8;
	
	static void initWeights (){
			
		for(int i =0; i< inputLayerLength; i++){
			for(int h=0; h<hiddenLayerLength; h++){
				inputHiddenWeights[i][h] = -0.1+ 0.2*Math.random();
			}
		}
				
		for(int h =0; h< hiddenLayerLength; h++){
			for(int i =0; i< outputLayerLength; i++){
				hiddenOutputWeights[h][i] = -0.1+ 0.2*Math.random();
			}
		}
		
		
	}
	
	static double sigmoid(double x){
		return 1.0 / (1+ Math.exp(-x));
	}
	
	static double dsigmoid(double x){
		return x* (1-x);
	}
	
static void getOutput(int currentInput){
		byte[] inputLayerValues = trainingSet[currentInput];
		for(int h = 0; h < hiddenLayerLength; h++){
			double dot =0;
			for(int i =0; i< inputLayerLength; i++){
				dot+= inputLayerValues[i]*inputHiddenWeights[i][h];
			}
			hiddenLayerValues[h] = sigmoid(dot);
		}
		
		
		for(int k = 0; k < outputLayerLength; k++){
			double output = 0;
			for(int h = 0; h<hiddenLayerLength; h++){
				output += hiddenLayerValues[h]*hiddenOutputWeights[h][k];
			}
			outputLayerValues[k]= sigmoid(output);
		}
				
	}
	
	

	
static void backpropagation (int patternIndex ){
	
	double[] outputDelta = new double [outputLayerLength];
			
	for(int k =0; k<outputLayerLength; k++){
		double error = expectedOutputs[patternIndex][k]-outputLayerValues[k];
		outputDelta[k] = error* dsigmoid(outputLayerValues[k]);
	}
	
	for(int k=0; k< outputLayerLength; k++){
		for(int h =0; h< hiddenLayerLength; h++){
			hiddenOutputWeights[h][k]+= hiddenLayerValues[h]*LEARNING_RATE * outputDelta[k];
		}
	}
	double[] hiddenDeltas = new double[hiddenLayerLength];
	
	for(int h =0; h< hiddenLayerLength; h++){
		double dotProduct =0;
		for(int k =0; k< outputLayerLength; k++)
			dotProduct += outputDelta[k] * hiddenOutputWeights[h][k];
			
	    hiddenDeltas[h] = dsigmoid(hiddenLayerValues[h])*dotProduct;
	}
	
	byte[] inputLayerValues = trainingSet[patternIndex];
	for(int i =0; i< inputLayerLength; i++){
		for(int h =0; h< hiddenLayerLength; h++){ 
			
			inputHiddenWeights[i][h] +=inputLayerValues[i]* LEARNING_RATE* hiddenDeltas[h];
		}
	}
	
}
	
 	static boolean isCorrect(int patternIndex){
 		for(int k=0; k< outputLayerLength; k++){
 			double diff = Math.abs(expectedOutputs[patternIndex][k]-outputLayerValues[k]);
 			if(diff>= TRAIN_THRESHOLD)
 				return false;
 			
 		}
 		
 		return true;
 	}

	
	
	
	static void readTrainPatterns(String filename) throws IOException {
		Scanner sc = new Scanner(new FileReader(filename));
		trainingSetSize     = sc.nextInt();
		inputLayerLength    = sc.nextInt();
		outputLayerLength   = sc.nextInt();
		hiddenLayerLength   = inputLayerLength / 2;
		trainingSet         = new byte[trainingSetSize][inputLayerLength];
		inputHiddenWeights  = new double[inputLayerLength][hiddenLayerLength];
		hiddenLayerValues   = new double[hiddenLayerLength];
		hiddenOutputWeights = new double[hiddenLayerLength][outputLayerLength];
		expectedOutputs     = new double[trainingSetSize][outputLayerLength];
	    outputLayerValues	= new double[outputLayerLength];
		for (int r = 0; r < trainingSetSize; r++) {
			for (int c = 0; c < inputLayerLength; c++) {
				trainingSet[r][c] = sc.nextByte();
			}
			for (int c = 0; c < outputLayerLength; c++) {
				expectedOutputs[r][c] = sc.nextDouble();
			}
		}
		sc.close();
	}
	
	static boolean testPattern(byte[] pattern, double[] expectedOutput) {
		for(int h = 0; h< hiddenLayerLength; h++){
			double dotProduct =0;
			for(int i =0; i<inputLayerLength; i++){
				dotProduct += pattern[i]* inputHiddenWeights[i][h];
			}
			hiddenLayerValues[h] = sigmoid(dotProduct);
		}
		
		for(int k = 0; k< outputLayerLength; k++){
			double dotProduct =0;
			for(int h = 0; h< hiddenLayerLength; h++){
				dotProduct+= hiddenLayerValues[h]*hiddenOutputWeights[h][k];
			}
			outputLayerValues[k]= sigmoid(dotProduct);
			
		}
		
		System.out.println(Arrays.toString(outputLayerValues)+ " ->" +
						Arrays.toString(expectedOutput));
		
		for(int k =0; k< outputLayerLength; k++){
			double diff = Math.abs(expectedOutput[k]-outputLayerValues[k]);
			if(diff >= TEST_THRESHOLD)
				return false;
		}
		
		return true;
	}
	
	static void testNN(String filename) throws IOException {
		Scanner sc = new Scanner(new FileReader(filename));
		int testingSetSize = sc.nextInt();
		int patternLength = sc.nextInt();
		int outputLength = sc.nextInt();
		byte[] testPattern = new byte[patternLength];
		double[] expectedOutput = new double[outputLength];
		int correctCount = 0;
		for (int r = 0; r < testingSetSize; r++) {
			for (int c = 0; c < patternLength; c++) {
				testPattern[c] = sc.nextByte();
			}
			for (int c = 0; c < outputLength; c++) {
				expectedOutput[c] = sc.nextDouble();
			}
			if(testPattern(testPattern, expectedOutput)) {
				correctCount ++;
				System.out.println( " .... YES");
			} else {
				System.out.println( " .... NO");
			}
		}
		sc.close();
		System.out.println(correctCount + " / " + testingSetSize);
	}
	
	static void trainNN() {
		initWeights();
		int corrects  =0;
		
		tag1: for(int epoch = 1; epoch <=100; epoch++){
			for(int patternIndex =0; patternIndex < trainingSetSize; patternIndex++){
				getOutput(patternIndex);
				if(isCorrect(patternIndex)){
					corrects++;
					if(corrects >= trainingSetSize)
						break tag1;
				}else{
					corrects=0;
					backpropagation(patternIndex);
				}
			}
			
		}
		
	}
	
	public static void main(String[] args) throws IOException {
		readTrainPatterns("trainingSet2.txt");
		trainNN();
		testNN("testingSet2.txt");
	}
}