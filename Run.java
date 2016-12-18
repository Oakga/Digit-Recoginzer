import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Created by root on 12/14/16.
 */
public class Run {

    public static void main(String args[]) {


        //intial values (set these values)
        int Trial = 100;
        Double hidden_bias = 0.35;
        Double output_bias = 0.2;
        Double Learning_rate = 0.57;
        int num_inputs = 784;
        int num_hidden = 500;
        int num_outputs = 1;

        Scanner infile = null;
        String data[];
        ArrayList<Double> total_errors = new ArrayList<Double>();
        ArrayList<Double> training_inputs = null;
        ArrayList<Double> training_outputs = null;

        try {
            infile = new Scanner(new File("train.csv"));
        } catch (IOException e) {
            sop("File not found");
        }

        //nn creation
        Neural_Network nn = new Neural_Network(Learning_rate,
                num_inputs,
                num_hidden,
                num_outputs,
                hidden_bias,
                output_bias
        );


        infile.nextLine();
        int count_trials=0;
        ArrayList<ArrayList<Double>> pixelList = new ArrayList<ArrayList<Double>>();
        while (infile.hasNextLine() && count_trials < 500) {


            data = infile.nextLine().split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)");
            pixelList = getNextRow(data);


            training_inputs = pixelList.get(0);
            training_outputs = pixelList.get(1);

            nn.train(training_inputs, training_outputs);
            nn.calculate_total_error(training_inputs, training_outputs);

            count_trials++;
        }
        infile.close();

    }

    //Helper Functions
    public static double round(double value) {
        return (double) Math.round(value * 10000d) / 10000d;
    }

    public static ArrayList<ArrayList<Double>> getNextRow(String pixels[]) {
        ArrayList<ArrayList<Double>> result = new ArrayList<ArrayList<Double>>();
        ArrayList<Double> output = new ArrayList<Double>();
        ArrayList<Double> input = new ArrayList<Double>();
        double max = -1, num = 0;
        output.add(new Double(pixels[0]));
        for (int a = 1; a < 785; a++) {
            num = Integer.parseInt(pixels[a]);
            if (num > max) max = num;
            input.add(new Double(pixels[a]));
        }

        //this for loop normalizes all the indexes

        for (int a = 1; a < input.size(); a++) {
            input.set(a, (input.get(a)) / max);
        }

        result.add(input);
        result.add(output);

        return result;
    }


    public static void sop(Object text) {
        System.out.println("" + text);
    }
}
