import java.util.ArrayList;

/**
 * Created by Oakga on 12/14/16.
 */
public class Neuron_Layer {
    Double bias;
    ArrayList<Neuron> neurons = new ArrayList<Neuron>();

    public Neuron_Layer(int num_of_neurons, Double bias) {
        this.bias = bias;
        set_neurons(num_of_neurons);
    }

    public void set_neurons(int num_of_neurons) {
        for (int i = 0; i < num_of_neurons; i++) {
            Neuron new_neuron = new Neuron(i + 1, bias);
            neurons.add(new_neuron);
        }
    }

    public void inspect() {
        int count = 0;
        sop("\nLayer inspection");
        sop("Number of Neurons: " + neurons.size() + "\n");
        for (Neuron n : neurons) {
            sop("Neuron id: " + n.id);
            count = 0;
            sop("Weights are as following: ");
            for (Double w : n.weights) {
                sop("Weight " + count + " : " + w);
                count++;
            }
            sop("Bias :" + n.bias);
        }
    }

    public ArrayList<Double> feed_forward(ArrayList<Double> input) {
        ArrayList<Double> outputs = new ArrayList<Double>();
        for (Neuron n : neurons) {
            outputs.add(n.calculate_output(input));
        }
        return outputs;
    }

    public ArrayList<Double> get_outputs() {
        ArrayList<Double> outputs = new ArrayList<Double>();
        for (Neuron n : neurons) {
            outputs.add(n.output);
        }
        return outputs;
    }

    public static double round(double value) {
        return (double) Math.round(value * 10000d) / 10000d;
    }

    public static void sop(Object text) {
        System.out.println("" + text);
    }


}
