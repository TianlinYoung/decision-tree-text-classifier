import java.util.*;
import java.util.function.*;
import java.io.*;

// Client class for interaction with Classifiers
public class Client {
    public static final String TRAIN_FILE = "data/emails/train.csv"; 
    public static final String TEST_FILE = "data/emails/test.csv";      
    
    // Index of the column (feature) we're trying to predict. In the case of Email files, 
    // index 0 corresponds with the first column: Category
    public static final int LABEL_INDEX = 0;

    // Index of the column (feature) we use to predict our label. In the case of Email files,
    // index 1 corresponds with the second column: Message
    public static final int CONTENT_INDEX = 1;

    public static void main(String[] args) throws FileNotFoundException {
        Scanner console = new Scanner(System.in);

        printBanner();
        System.out.println("Welcome to the CSE 123 Classifier!");
        System.out.println("(Remember to edit the TRAIN_FILE and TEST_FILE class constants " +
            "if you want to change the files being used!)");
        System.out.println();

        System.out.println("To begin, enter your desired mode of operation:");
        System.out.println();
        Classifier c = createModel(console);

        System.out.println();
        System.out.println("What would you like to do with your model?");
        int choice = -1;
        while (choice != 4) {
            System.out.println();
            System.out.println("1) Test with an input file (classify)");
            System.out.println("2) Get testing accuracy (calculateAccuracy)");
            System.out.println("3) Save classification tree (save)");
            System.out.println("4) Quit");
            System.out.print("Enter your choice here: ");

            choice = console.nextInt();
            while (choice != 1 && choice != 2 && choice != 3 && choice != 4) {
                System.out.print("Please enter a valid option from above: ");
                choice = console.nextInt();
            }

            if (choice == 1) {
                System.out.println("Please enter the path to the file you'd like to test");
                System.out.println("Example: \"./data/emails/test.csv\"");
                System.out.print("File path: ");
                String path = console.next();
                if (path.charAt(0) == '\"' && path.charAt(path.length() - 1) == '\"') {
                    path = path.substring(1, path.length() - 1);
                }
                evalModel(c, path);
            } else if (choice == 2) {
                testModel(c, TEST_FILE);
            } else if (choice == 3) {
                
                System.out.println("Would you like to save to a file or output the " +
                    "classification tree to console?");
                System.out.println("1) Save to a file");
                System.out.println("2) Output classification tree to console");
                choice = console.nextInt();
                if (choice == 1) {
                    System.out.print("Please enter the file name you'd like to save to: ");
                    c.save(new PrintStream(console.next() + ".txt"));
                } else {
                    System.out.println();
                    System.out.println("Save output:");
                    System.out.println();
                    c.save(System.out);
                }
            }
        }
    }

    // Creates a classifier from a client provided information by either:
    //      Loading a previously created model file or
    //      Training a model from a provided dataset
    // Requires a Scanner connected to the console to retrieve user input
    // Throws a FileNotFoundException
    //      If one of the client provided files doesn't exist
    private static Classifier createModel(Scanner console) throws FileNotFoundException {
        System.out.println("1) Train classification model (Two List Constructor)");
        System.out.println("2) Load model from file (Scanner Constructor)");
        System.out.print("Enter your choice here: ");

        int choice = console.nextInt();
        while (choice != 1 && choice != 2) {
            System.out.print("Please enter a valid option from above: ");
            choice = console.nextInt();
        }

        if (choice == 1) {
            System.out.println();
            System.out.println("Would you like to shuffle the data?");
            System.out.println("1) Yes (Recommended for testing finalized models)");
            System.out.println("2) No  (Recommended for debugging models)");
            choice = console.nextInt();
            if (choice == 1) {
                DataLoader loader = new DataLoader(TRAIN_FILE, LABEL_INDEX, CONTENT_INDEX, true);
                return new Classifier(loader.getData(), loader.getLabels());
            } else {
                DataLoader loader = new DataLoader(TRAIN_FILE, LABEL_INDEX, CONTENT_INDEX, false);
                return new Classifier(loader.getData(), loader.getLabels());
            }
        } else {
            System.out.println("Please enter the path to the file you'd like to load");
            System.out.println("Example: \"./trees/simple.txt\"");
            System.out.print("File path: ");
            String path = console.next();
            if (path.charAt(0) == '\"' && path.charAt(path.length() - 1) == '\"') {
                path = path.substring(1, path.length() - 1);
            }
            Scanner input = new Scanner(new File(path));
            return new Classifier(input);
        }
    }

    // Uses the given Classifier to predict labels for the datapoints within the given testing 
    //      file, printing out the results
    // Throws a FileNotFoundException
    //      If the provided testing dataset file doesn't exist
    private static void evalModel(Classifier c, String fileName) throws FileNotFoundException {
        DataLoader loader = new DataLoader(fileName, LABEL_INDEX, CONTENT_INDEX, false);
        List<String> results = new ArrayList<>();
        for (TextBlock data : loader.getData()) {
            results.add(c.classify(data));
        }
        System.out.println("Results: " + results);
    }

    // Tests the given Classifier on the datapoints within the given testing file, printing out the
    //      accuracies for labels encountered during testing
    // Throws a FileNotFoundException
    //      If the provided testing dataset file doesn't exist
    private static void testModel(Classifier c, String fileName) throws FileNotFoundException {
        DataLoader loader = new DataLoader(fileName, LABEL_INDEX, CONTENT_INDEX, false);
        Map<String, Double> labelToAccuracy = c.calculateAccuracy(loader.getData(),
                                                                  loader.getLabels());
        for (String label : labelToAccuracy.keySet()) {
            System.out.println(label + ": " + labelToAccuracy.get(label));
        }
    }

    // Prints the Classifier banner to console
    private static void printBanner() {
        System.out.println(" _______  ___      _______  _______  _______  ___  " +
            " _______  ___   _______  ______   ");
        System.out.println("|       ||   |    |   _   ||       ||       ||   | " +
            "|       ||   | |       ||    _ |  ");
        System.out.println("|       ||   |    |  |_|  ||  _____||  _____||   | " +
            "|    ___||   | |    ___||   | ||  ");
        System.out.println("|       ||   |    |       || |_____ | |_____ |   | " +
            "|   |___ |   | |   |___ |   |_||_ ");
        System.out.println("|      _||   |___ |       ||_____  ||_____  ||   | " +
            "|    ___||   | |    ___||    __  |");
        System.out.println("|     |_ |       ||   _   | _____| | _____| ||   | " +
            "|   |    |   | |   |___ |   |  | |");
        System.out.println("|_______||_______||__| |__||_______||_______||___| " +
            "|___|    |___| |_______||___|  |_|");
    }

}
