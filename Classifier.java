// Tianlin Yang
// 03/04/2026
// CSE 123
// P3: Cornbear's Classifier
// TA: Aidan Jefferson Suen

import java.io.*;
import java.util.*;

// This class represents a binary decision-tree text classifier.
// It can be constructed either by reading a saved model from a Scanner or by training from
// labeled TextBlock data, then used to classify new TextBlocks and save the trained model.
public class Classifier {
    private ClassifierNode overallRoot;


    // Behavior: Constructs a classifier by reading a previously-saved decision tree from 'input'.
    //           The resulting tree becomes the model used for classification.
    // Exceptions: IllegalArgumentException if input is null.
    //             IllegalStateException if input has no lines (empty model).
    //             IllegalArgumentException if the tree description ends before a required line is
    //             read, or if a "Feature: ..." line is not followed by a "Threshold: ..." line.
    // Parameters: input - the Scanner providing the saved tree text.
    public Classifier(Scanner input) {
        if (input == null) {
            throw new IllegalArgumentException();
        }
        if (!input.hasNextLine()) {
            throw new IllegalStateException();
        }
        
        overallRoot = readTree(input);
    }


    // Behavior: Reads and returns the root of a classifier tree from 'input'. Internal nodes are
    //           described by a "Feature: ..." line followed by a "Threshold: ..." line, then the
    //           left subtree and right subtree, in pre-order (left then right). Leaf nodes are
    //           described by a single label line.
    // Exceptions: IllegalArgumentException if the tree description ends before a required line is
    //             read, or if a "Feature: ..." line is not followed by a "Threshold: ..." line.
    // Returns: a ClassifierNode representing the root of the parsed tree.
    // Parameters: input - the Scanner to read the tree from.
    private static ClassifierNode readTree(Scanner input) {
        if (!input.hasNextLine()) {
            throw new IllegalArgumentException();
        }

        String featureLine = input.nextLine();

        if (featureLine.startsWith("Feature: ")) {
            String feature = featureLine.substring("Feature: ".length());

            if (!input.hasNextLine()) {
                throw new IllegalArgumentException();
            }
            String thresholdLine = input.nextLine();
            if (!thresholdLine.startsWith("Threshold: ")) {
                throw new IllegalArgumentException();
            }

            double threshold = Double.parseDouble(thresholdLine.substring("Threshold: ".length()));

            ClassifierNode left = readTree(input);
            ClassifierNode right = readTree(input);
            return new ClassifierNode(feature, threshold, left, right);
        }

        return new ClassifierNode(featureLine, null);
    }


    // Behavior: Constructs a classifier by training a decision tree from the provided labeled
    //           data. Starts the tree with the first example as a leaf, then incorporates
    //           remaining examples.
    // Exceptions: IllegalArgumentException if data or labels is null, if they have different
    //             sizes, or if data is empty.
    // Parameters: data - the training TextBlocks.
    //             labels - the expected label for each TextBlock in data.
    public Classifier(List<TextBlock> data, List<String> labels) {
        if (data == null || labels == null || data.size() != labels.size() || data.isEmpty()) {
            throw new IllegalArgumentException();
        }

        overallRoot = new ClassifierNode(labels.get(0), data.get(0));
        trainFromIndex(1, data, labels);
    }


    // Behavior: Trains the model on all examples in data/labels starting at 'index',
    //           updating overallRoot as each example is incorporated.
    // Parameters: index  - the index of the next example to train on.
    //             data - the training TextBlocks.
    //             labels - the expected labels for each TextBlock.
    private void trainFromIndex(int index, List<TextBlock> data, List<String> labels) {
        if (index >= data.size()) {
            return;
        }
        overallRoot = trainOne(overallRoot, data.get(index), labels.get(index));
        trainFromIndex(index + 1, data, labels);
    }


    // Behavior: Incorporates (input, expectedLabel) into the classifier subtree rooted at 'curr'.
    //           If curr is null, creates a new leaf. If curr is a leaf with a different label,
    //           converts it into an internal decision node that separates the old and new examples
    //           using the feature with the biggest difference and a midpoint threshold. Otherwise,
    //           trains into the appropriate child subtree.
    // Returns: the (possibly new) root of the updated subtree after training on this example.
    // Parameters: curr - the current subtree root to train into.
    //             input - the TextBlock to add to the model.
    //             expectedLabel - the correct label for input.
    private static ClassifierNode trainOne(ClassifierNode curr, TextBlock input, String expectedLabel) {
        if (curr == null) {
            return new ClassifierNode(expectedLabel, input);
        }

        if (curr.isLeaf()) {
            if (curr.label.equals(expectedLabel)) {
                return curr;
            }

            TextBlock oldData = curr.data;

            String feature = input.findBiggestDifference(oldData);
            double threshold = midpoint(input.get(feature), oldData.get(feature));

            ClassifierNode newLeaf = new ClassifierNode(expectedLabel, input);

            if (input.get(feature) < threshold) {
                return new ClassifierNode(feature, threshold, newLeaf, curr);
            } else {
                return new ClassifierNode(feature, threshold, curr, newLeaf);
            }
        }

        if (input.get(curr.feature) < curr.threshold) {
            curr.left = trainOne(curr.left, input, expectedLabel);
        } else {
            curr.right = trainOne(curr.right, input, expectedLabel);
        }
        return curr;
    }


    // Behavior: Classifies the given TextBlock using this classifier's decision tree.
    // Exceptions: IllegalArgumentException if input is null.
    // Returns: the predicted label for the provided TextBlock.
    // Parameters: input - the TextBlock to classify.
    public String classify(TextBlock input) {
        if (input == null) {
            throw new IllegalArgumentException();
        }
        return classify(overallRoot, input);
    }


    // Behavior: Traverses the classifier tree rooted at 'curr' to classify 'input'.
    //           At internal nodes, compares input's feature value to the node's threshold to
    //           choose left (< threshold) or right (>= threshold). At a leaf, returns the
    //           stored label.
    // Returns: the predicted label for input according to the subtree rooted at curr.
    // Parameters: curr  - the current node in the classifier tree.
    //             input - the TextBlock being classified.
    private static String classify(ClassifierNode curr, TextBlock input) {
        if (curr.isLeaf()) {
            return curr.label;
        }

        if (input.get(curr.feature) < curr.threshold) {
            return classify(curr.left, input);
        } else {
            return classify(curr.right, input);
        }
    }


    // Behavior: Writes this classifier's decision tree to 'output' in the same format expected by
    //           the Scanner-based constructor, so the model can be reloaded later.
    // Exceptions: IllegalArgumentException if output is null.
    // Parameters: output - the PrintStream to write the model to.
    public void save(PrintStream output) {
        if (output == null) {
            throw new IllegalArgumentException();
        }
        save(overallRoot, output);
    }


    // Behavior: Writes the subtree rooted at 'curr' to 'output'. Leaf nodes are
    //           written as a single label line. Internal nodes are written as "Feature: ...",
    //           then "Threshold: ...", followed by the left subtree then the right subtree.
    // Parameters: curr - the current node to save.
    //             output - the PrintStream to write to.
    private void save(ClassifierNode curr, PrintStream output) {
        if (curr == null) {
            return;
        }

        if (curr.isLeaf()) {
            output.println(curr.label);
        } else {
            output.println("Feature: " + curr.feature);
            output.println("Threshold: " + curr.threshold);
            save(curr.left, output);
            save(curr.right, output);
        }
    }

    // This class represents a node in the classifier decision tree. Leaf nodes store a label and
    // its associated training TextBlock, while internal nodes store a feature, a threshold, and
    // links to left/right children used to route classification and training.
    private static class ClassifierNode {
        public String label;
        public TextBlock data;

        public String feature;
        public double threshold;
        public ClassifierNode left;
        public ClassifierNode right;


        // Behavior: Constructs a leaf node containing the provided label and associated training
        //           data.
        // Parameters: label - the label stored at this leaf.
        //             data - the TextBlock associated with this label.
        public ClassifierNode(String label, TextBlock data) {
            this.label = label;
            this.data = data;
        }


        // Behavior: Constructs an internal decision node that splits on the given feature and
        //           threshold.
        // Parameters: feature - the feature name used for the decision at this node.
        //             threshold - the cutoff value used to choose between children.
        //             left - the child subtree for values less than threshold.
        //             right - the child subtree for values greater than or equal to threshold.
        public ClassifierNode(String feature, double threshold,
                            ClassifierNode left, ClassifierNode right) {
            this.feature = feature;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }


        // Behavior: Determines whether this node is a leaf node.
        // Returns: true if this node stores a label (leaf); false if it is an internal decision
        //          node.
        public boolean isLeaf() {
            return label != null;
        }
    }


    ////////////////////////////////////////////////////////////////////
    // PROVIDED METHODS - **DO NOT MODIFY ANYTHING BELOW THIS LINE!** //
    ////////////////////////////////////////////////////////////////////

    // Helper method to calculate the midpoint of two provided doubles.
    private static double midpoint(double one, double two) {
        return Math.min(one, two) + (Math.abs(one - two) / 2.0);
    }

    // Behavior: Calculates the accuracy of this model on provided Lists of 
    //           testing 'data' and corresponding 'labels'. The label for a 
    //           datapoint at an index within 'data' should be found at the 
    //           same index within 'labels'.
    // Exceptions: IllegalArgumentException if the number of datapoints doesn't match the number 
    //             of provided labels
    // Returns: a map storing the classification accuracy for each of the encountered labels when
    //          classifying
    // Parameters: data - the list of TextBlock objects to classify. Should be non-null.
    //             labels - the list of expected labels for each TextBlock object. 
    //             Should be non-null.
    public Map<String, Double> calculateAccuracy(List<TextBlock> data, List<String> labels) {
        // Check to make sure the lists have the same size (each datapoint has an expected label)
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException(
                    String.format("Length of provided data [%d] " +
                                    "doesn't match provided labels [%d]", 
                            data.size(), labels.size()));
        }

        // Create our total and correct maps for average calculation
        Map<String, Integer> labelToTotal = new HashMap<>();
        Map<String, Double> labelToCorrect = new HashMap<>();
        labelToTotal.put("Overall", 0);
        labelToCorrect.put("Overall", 0.0);

        for (int i = 0; i < data.size(); i++) {
            String result = classify(data.get(i));
            String label = labels.get(i);

            // Increment totals depending on resultant label
            labelToTotal.put(label, labelToTotal.getOrDefault(label, 0) + 1);
            labelToTotal.put("Overall", labelToTotal.get("Overall") + 1);
            if (result.equals(label)) {
                labelToCorrect.put(result, labelToCorrect.getOrDefault(result, 0.0) + 1);
                labelToCorrect.put("Overall", labelToCorrect.get("Overall") + 1);
            }
        }

        // Turn totals into accuracy percentage
        for (String label : labelToCorrect.keySet()) {
            labelToCorrect.put(label, labelToCorrect.get(label) / labelToTotal.get(label));
        }

        return labelToCorrect;
    }
}
