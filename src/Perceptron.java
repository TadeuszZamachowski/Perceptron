import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Perceptron {

    private double _bias;
    private double _learningRate;
    private List<Double> _weights;

    public Perceptron(double bias, double learningRate, int size) {
        _bias = bias;
        _learningRate = learningRate;
        _weights = initializeWeights(size);
    }

    public int calculateOutput(List<Double> inputs) {
        double net = scalarProduct(inputs, _weights) - _bias;
        if (net >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

    public void updateWeightsAndBias(List<Double> inputs, int y, double d) {
        _weights = deltaRuleForVectors(_weights,_learningRate,d,y,inputs);
        _bias = _bias - _learningRate*(d - y);
    }

    public void normalizeWeightsVector() {
        for (int i = 0; i < _weights.size(); i++) {
            double change = _weights.get(i)/_weights.size();
            _weights.set(i, change);
        }
    }


    public void train(List<Double> inputs, double target) {
        int output = calculateOutput(inputs);

        this.updateWeightsAndBias(inputs,output,target);
        //??????????????????????????????
        //this.normalizeWeightsVector();
    }

    public static List<Double> initializeWeights(int size) {
        List<Double> weights = new ArrayList<>();
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            double value = round(r.nextDouble()+1,1);
            weights.add(value);
        }
        return weights;
    }

    public static double scalarProduct(List<Double> l1, List<Double> l2) {
        if (l1.size() != l2.size()) {
            return -1;
        } else {
            double result = 0;
            for (int i = 0; i < l1.size(); i++) {
                result += l1.get(i) * l2.get(i);
            }
            return result;
        }
    }

    public static List<Double> deltaRuleForVectors(List<Double> weights, double learningRate, double d, int y, List<Double> input) {
        double multiplier = learningRate*(d-y);
        // scaling X vector
        for (int i = 0; i < input.size(); i++) {
            double change = input.get(i) * multiplier;
            input.set(i,change);
        }
        // adding W and X
        for (int i = 0; i < weights.size(); i++) {
            double change = weights.get(i) + input.get(i);
            weights.set(i, change);
        }
        return weights;
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
