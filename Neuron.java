import java.util.ArrayList;

/**
 * Created by Oakga on 12/14/16.
 */
public class Neuron {
    int id;
    Double bias;
    ArrayList<Double> inputs = new ArrayList<Double>();
    ArrayList<Double> weights = new ArrayList<Double>();
    Double output;

    public Neuron(int id, Double bias) {
        this.id = id;
        this.bias = bias;
    }

    public Double calculate_output(ArrayList<Double> inputs) {
        this.inputs = inputs;
        this.output = squash(calculate_total_net_input());
        return this.output;
    }

    public Double calculate_total_net_input() {
        Double total = 0.0;
        int count = 0;
        for (Double i : inputs) {
            total += i * weights.get(count);
            count++;
        }
        return total + this.bias;
    }

    public double squash(Double total_net_input) {
        return (1 / (1 + Math.pow(Math.E, (-1 * total_net_input))));
    }

    //calculating errors functions

    public Double calculate_error(Double target_output) {
        return 0.5 * Math.pow((target_output - this.output), 2);
    }

    public Double calculate_pd_error_wrt_total_net_input(Double target_output) {
        return calculate_pd_error_wrt_output(target_output) * calculate_pd_total_net_wrt_input();
    }

    public Double calculate_pd_error_wrt_output(Double target_output) {
        return -(target_output - this.output);
    }

    public Double calculate_pd_total_net_wrt_input() {
        return this.output * (1 - this.output);
    }

    public Double calculate_pd_total_net_input_wrt_weight(int index) {
        return this.inputs.get(index);
    }

    //Helper Functions
    public static double round(double value) {
        return (double) Math.round(value * 10000d) / 10000d;
    }

    public static void sop(Object text) {
        System.out.println("" + text);
    }
}
