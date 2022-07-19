import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        File trainingSet = new File("data.csv");
        List<List<String>> trainData = readCsvFile(trainingSet);

        File testingSet = new File("testdata.csv");
        List<List<String>> testData = readCsvFile(testingSet);

        List<List<Double>> procTrainData = processData(trainData);
        List<List<Double>> procTestData = processData(testData);



        double bias = 3;
        double learningRate = 0.01;
        int size = trainData.get(0).size() -1;

        Perceptron perceptron = new Perceptron(bias,learningRate, size);

        trainPerceptron(perceptron, procTrainData);
        testPerceptron(perceptron, procTestData);

        System.out.println("Input your data:");
        System.out.println("[v1,v2,...,vn]");

        Scanner sc = new Scanner(System.in);

        while (true) {
            String input = sc.nextLine();
            perceptronInput(perceptron, input);
        }

    }


    public static void trainPerceptron(Perceptron perceptron, List<List<Double>> data) {
        double totalSize = data.size();
        double currentAccuracy = 0;
        int numOfIterations = 0;
        double globalError = totalSize;

        while (globalError >= 3) {
            globalError = 0;
            for (int i = 0; i < data.size(); i++) {
                List<Double> vectors = new ArrayList<>();
                double d = 0;
                for (int j = 0; j < data.get(i).size(); j++) {
                    if (j == data.get(i).size() - 1) {
                        d = data.get(i).get(j);
                    } else {
                        vectors.add(data.get(i).get(j));
                    }
                }
                int y = perceptron.calculateOutput(vectors);
                double localError = (d-y)*(d-y);
                perceptron.train(vectors, d);
                globalError += localError;
            }
            currentAccuracy = (totalSize -globalError)/totalSize*100;
            numOfIterations++;
            System.out.println("Iteration "+numOfIterations+" Accuracy "+ currentAccuracy+"%");
        }
    }

    public static void testPerceptron(Perceptron perceptron, List<List<Double>> data) {
        double totalTestSize = data.size();
        double testAccuracy;
        double testCorrect = 0;

        for (int i = 0; i < data.size(); i++) {
            List<Double> vectors = new ArrayList<>();
            double d = 0;
            for (int j = 0; j < data.get(i).size(); j++) {
                if (j == data.get(i).size() - 1) {
                    d = data.get(i).get(j);
                } else {
                    vectors.add(data.get(i).get(j));
                }
            }
            int y = perceptron.calculateOutput(vectors);
            if (y == d) {
                testCorrect++;
            }
        }
        testAccuracy = testCorrect/totalTestSize * 100;
        System.out.println("Test accuracy = "+testAccuracy+"%");
    }

    public static void perceptronInput(Perceptron perceptron, String input) {
        String[] data = input.split(",");
        List<Double> vectors = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            vectors.add(Double.parseDouble(data[i]));
        }
        int y = perceptron.calculateOutput(vectors);
        if (y == 1) {
            System.out.println("This data matches Iris-versicolor");
        } else {
            System.out.println("This data matches Iris-virginica");
        }
    }


    public static List<List<Double>> processData(List<List<String>> rawData) {
        List<List<Double>> processedData = new ArrayList<>();

        String name = rawData.get(0).get(rawData.get(0).size() - 1);

        for (int i = 0; i< rawData.size(); i++) {
            List<Double> entry = new ArrayList<>();
            for (int j = 0; j < rawData.get(i).size(); j++) {
                if (j == rawData.get(i).size() -1) {
                    if (rawData.get(i).get(j).equals(name)) {
                        entry.add(1.0);
                    } else  {
                        entry.add(0.0);
                    }
                } else {
                    entry.add(Double.parseDouble(rawData.get(i).get(j)));
                }
            }
            processedData.add(entry);
        }
        return processedData;
    }



    public static List<List<String>> readCsvFile(File file) throws IOException {
        List<List<String>> list = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(file));

        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            list.add(new ArrayList(Arrays.asList(values)));
        }

        return list;
    }

}
