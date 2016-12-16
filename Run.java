import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by root on 12/14/16.
 */
public class Run {

    public static void main(String args[]) {


        //intial values
        Double hidden_bias = 0.35;
        Double output_bias = .6;

        //hidden layer
        Double[] first_hidden_neuron = {0.15, 0.2};
        Double[] second_hidden_neuron = {0.25, 0.3};
        ArrayList<Double[]> hidden_layer_weights = new ArrayList<Double[]>();
        hidden_layer_weights.add(first_hidden_neuron);
        hidden_layer_weights.add(second_hidden_neuron);

        //output layer
        Double[] first_output_neuron = {0.4, 0.45};
        Double[] second_output_neuron = {0.5, 0.55};
        ArrayList<Double[]> output_layer_weights = new ArrayList<Double[]>();
        output_layer_weights.add(first_output_neuron);
        output_layer_weights.add(second_output_neuron);

        //nn creation
        Neural_Network nn = new Neural_Network(0.3,
                2,
                2,
                2,
                hidden_layer_weights,
                output_layer_weights,
                hidden_bias,
                output_bias
        );

        sop("Net initialization");
        nn.hidden_layer.inspect();
        nn.output_layer.inspect();

        //train
        ArrayList<Double> training_inputs = new ArrayList<Double>(Arrays.asList(0.05, 0.1));
        ArrayList<Double> training_outputs = new ArrayList<Double>(Arrays.asList(0.01, 0.99));

        ArrayList<Double> total_errors= new ArrayList<Double>();
        Double total_error;
        sop("Training executing");
        int count =0;
        while(count<100000) {
            sop("Trial: "+count+" starting");
            nn.train(training_inputs, training_outputs);
            nn.inspect();

            total_error=nn.calculate_total_error(training_inputs,training_outputs);
            total_errors.add(total_error);
            sop("Trail "+count+" total error: "+total_error);
            count++;
        }

        //Printing out total errors this will be less in each case if correct
        sop("Analyzing total errors in each trails");
        for(int i=0;i<total_errors.size();i++){
            sop("Trail "+i+" total error: "+total_errors.get(i));
        }


    }

    //Helper Functions
    public static double round(double value) {
        return (double) Math.round(value * 10000d) / 10000d;
    }

    public static void sop(Object text) {
        System.out.println("" + text);
    }
}
