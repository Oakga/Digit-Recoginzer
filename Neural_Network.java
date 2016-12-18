import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;

/**
 * Created by Oakga on 12/14/16.
 */
public class Neural_Network {
    Double Learning_rate = 0.5;
    int num_inputs;
    Neuron_Layer hidden_layer;
    Neuron_Layer output_layer;
    ArrayList<Num> result;
    int right = 0;
    int wrong = 0;

    public Neural_Network(
            Double Learning_rate,
            int num_inputs,
            int num_hidden,
            int num_outputs,
            ArrayList<Double[]> hidden_layer_weights,
            ArrayList<Double[]> output_layer_weights,
            Double hidden_layer_bias,
            Double output_layer_bias
    ) {
        this.Learning_rate = Learning_rate;
        this.num_inputs = num_inputs;
        this.hidden_layer = new Neuron_Layer(num_hidden, hidden_layer_bias);
        this.output_layer = new Neuron_Layer(num_outputs, output_layer_bias);

        //hidden_layer_weights
        init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights);
        init_weights_from_hidden_layer_to_output_layer_neurons(output_layer_weights);

        //intialize number for results
        result = new ArrayList<Num>();
        intialize_result_Array_list(result);
    }

    public Neural_Network(
            Double Learning_rate,
            int num_inputs,
            int num_hidden,
            int num_outputs,
            Double hidden_layer_bias,
            Double output_layer_bias
    ) {
        this.Learning_rate = Learning_rate;
        this.num_inputs = num_inputs;
        this.hidden_layer = new Neuron_Layer(num_hidden, hidden_layer_bias);
        this.output_layer = new Neuron_Layer(num_outputs, output_layer_bias);

        //hidden_layer_weights
        init_weights_from_inputs_to_hidden_layer_neurons();
        init_weights_from_hidden_layer_to_output_layer_neurons();

        //intialize number for results
        result = new ArrayList<Num>();
        intialize_result_Array_list(result);
    }

    public ArrayList<Double> feed_forward(ArrayList<Double> inputs) {
        ArrayList<Double> hidden_layer_outputs = this.hidden_layer.feed_forward(inputs);
        ArrayList<Double> outputs;
        output_layer.output_turn=true;
        outputs = this.output_layer.feed_forward(hidden_layer_outputs);
        output_layer.output_turn=false;

        return outputs;
    }


    //test this out
    public void train(
            ArrayList<Double> training_inputs,
            ArrayList<Double> training_outputs
    ) {
        feed_forward(training_inputs);
        Double delta_wrt_output_error = 0.0;
        Double delta_wrt_hidden_error = 0.0;

        //output neuron preparation for gradient descent (delta) calculation
        int num_output_layer_neurons = output_layer.neurons.size();

        ArrayList<Double> pd_errors_wrt_output_neuron_total_net_input = new ArrayList<Double>();
        intialize_Array_list(pd_errors_wrt_output_neuron_total_net_input, num_output_layer_neurons);

        for (int i = 0; i < num_output_layer_neurons; i++) {
            delta_wrt_output_error = output_layer.neurons.get(i).calculate_pd_error_wrt_total_net_input(training_outputs.get(i));
            int j = pd_errors_wrt_output_neuron_total_net_input.size();
            pd_errors_wrt_output_neuron_total_net_input.set(i, delta_wrt_output_error);
        }

        //hidden neuron preparation for gradient descent (delta) calculation
        int num_hidden_layer_neurons = hidden_layer.neurons.size();

        ArrayList<Double> pd_errors_wrt_hidden_neuron_total_net_input = new ArrayList<Double>();
        intialize_Array_list(pd_errors_wrt_hidden_neuron_total_net_input, num_hidden_layer_neurons);

        for (int i = 0; i < num_hidden_layer_neurons; i++) {
            delta_wrt_hidden_error = 0.0;

            for (int j = 0; j < num_output_layer_neurons; j++) {
                delta_wrt_hidden_error += pd_errors_wrt_output_neuron_total_net_input.get(j) * output_layer.neurons.get(j).weights.get(i);
            }

            Double value = delta_wrt_hidden_error * hidden_layer.neurons.get(i).calculate_pd_total_net_wrt_input();
            pd_errors_wrt_hidden_neuron_total_net_input.set(i, value);
        }

        //update output neurons weights
        Double pd_error_wrt_weight = 0.0;
        for (int i = 0; i < num_output_layer_neurons; i++) {
            for (int j = 0; j < output_layer.neurons.get(i).weights.size(); j++) {
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input.get(i) * hidden_layer.neurons.get(i).calculate_pd_total_net_input_wrt_weight(j);
                Double new_weight = output_layer.neurons.get(i).weights.get(j) - Learning_rate * pd_error_wrt_weight;
                output_layer.neurons.get(i).weights.set(j, new_weight);
            }
        }

        //update hidden neuron weights
        for (int i = 0; i < num_hidden_layer_neurons; i++) {
            for (int j = 0; j < hidden_layer.neurons.get(i).weights.size(); j++) {
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input.get(i) * hidden_layer.neurons.get(i).calculate_pd_total_net_input_wrt_weight(j);
                Double new_weight = hidden_layer.neurons.get(i).weights.get(i) - Learning_rate * pd_error_wrt_weight;
                hidden_layer.neurons.get(i).weights.set(j, new_weight);
            }
        }


    }

    public Double calculate_total_error(ArrayList<Double> training_inputs, ArrayList<Double> training_outputs) {
        Double total_error = 0.0;

        ArrayList<Double> predicted_outputs = feed_forward(training_inputs);
        //update the digit array for evaluation
        update_result(training_outputs,predicted_outputs);

        for (int i = 0; i < training_outputs.size(); i++) {
            total_error += output_layer.neurons.get(i).calculate_error(training_outputs.get(i));
        }
        return total_error;
    }

    //Modified to only work with 1 label output
    private void update_result(ArrayList<Double> training_output, ArrayList<Double> predicted_outputs) {
        Double predicted_output = (predicted_outputs.get(0));
        Double actual_output = (training_output.get(0));

/*        sop("actual output: " + actual_output);
        sop("predicted output: " + predicted_output);*/

       /* Num digit_number_actual = result.get(actual_output);
        Num digit_number_predicted = result.get(predicted_output);*/

        /*digit_number_actual.show_up++;

        //For updating true positive and true negatives
        if (predicted_output == actual_output) {
            right++;
            sop("update true positive " + digit_number_predicted.number);
            sop("about to be updated digit number " + digit_number_predicted.number);
            digit_number_predicted.true_positive++;
            update_true_negative_for_all_other_digits(digit_number_predicted);
        }

        //For updating false negative and false positive
        else {
            wrong++;
            digit_number_predicted.false_negative++;
            digit_number_actual.false_positive++;
        }
*/
    }

    private void update_true_negative_for_all_other_digits(Num digit_number) {
        for (int i = 0; i < result.size(); i++) {
            if (i != digit_number.number) {
                result.get(i).true_negative++;
            }
        }
    }


    //Filler Functions

    //tested
    public void init_weights_from_inputs_to_hidden_layer_neurons() {
        Double weight_num = 0.0;
        int count = 0;
        for (Neuron h : hidden_layer.neurons) {
            for (int i = 0; i < num_inputs; i++) {
                hidden_layer.neurons.get(count).weights.add(rand(0.0, 1.0));
            }
            count++;
        }
    }

    //tested
    public void init_weights_from_hidden_layer_to_output_layer_neurons() {
        Double weight_num = 0.0;
        int count = 0;
        for (Neuron h : output_layer.neurons) {
            for (int i = 0; i < hidden_layer.neurons.size(); i++) {
                output_layer.neurons.get(count).weights.add(rand(0.0, 1.0));
            }
            count++;
        }
    }

    //tested
    public void init_weights_from_inputs_to_hidden_layer_neurons(ArrayList<Double[]> hidden_layer_weights) {
        Double weight_num = 0.0;
        int count = 0;
        for (Neuron h : hidden_layer.neurons) {
            for (int i = 0; i < num_inputs; i++) {
                hidden_layer.neurons.get(count).weights.add(hidden_layer_weights.get(count)[i]);
            }
            count++;
        }
    }

    //tested
    public void init_weights_from_hidden_layer_to_output_layer_neurons(ArrayList<Double[]> output_layer_weights) {
        Double weight_num = 0.0;
        int count = 0;
        for (Neuron h : output_layer.neurons) {
            for (int i = 0; i < hidden_layer.neurons.size(); i++) {
                output_layer.neurons.get(count).weights.add(output_layer_weights.get(count)[i]);
            }
            count++;
        }
    }

    //Helper Functions
    public static double round(double value) {
        return (double) Math.round(value * 100d) / 100d;
    }

    public static void sop(Object text) {
        System.out.println("" + text);
    }

    public static Double rand(double rangeMax, double rangeMin) {
        Random r = new Random();
        double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        return round(randomValue);
    }

    public static void intialize_Array_list(ArrayList<Double> list, int size) {
        for (int i = 0; i < size; i++) {
            list.add(0.0);
        }
    }

    public static void intialize_result_Array_list(ArrayList<Num> list) {
        for (int i = 0; i < 10; i++) {
            Num new_number = new Num(i);
            list.add(new_number);
        }
    }

    public void inspect_result() {
        for (int i = 0; i < result.size(); i++) {
            result.get(i).inspect();
        }
        sop("rights: " + right);
        sop("wrongs:" + wrong);
    }

    public void inspect() {
        sop("Inspecting the neural network");
        sop("Hidden Layer");
        hidden_layer.inspect();
        sop("Output Layer");
        output_layer.inspect();
    }
}
