import java.util.*;
import java.util.function.*;
import java.io.*;

// This class represents a DataLoader capable of loading both data and labels from
// A provided CSV file and manages them accordingly
public class DataLoader {
    private List<TextBlock> data;
    private List<String> labels;

    // Constructs a new DataLoader storing from the given file, where labels
    //      are taken from the given index, using the given 'contentIndex' to convert a
    //      particular row into the desired datapoint. Additionally, can choose to shuffle
    //      the data in the file.
    // 'filePath' should be non-null.
    // Throws a FileNotFoundException
    //      If the provided file doesn't exist
    public DataLoader(String filePath, int labelIndex, int contentIndex, boolean shuffle)
                      throws FileNotFoundException {
        this.data = new ArrayList<>();
        this.labels = new ArrayList<>();
        
        List<List<String>> rows = CsvReader.read(filePath);
        for (List<String> row : rows) {
            this.data.add(new TextBlock(row.get(contentIndex)));
            this.labels.add(row.get(labelIndex));
        }
        if (shuffle) {
            DataLoader.shuffle(this);
        }
    }

    // Returns the List of TextBlock data points currently stored by this DataLoader
    public List<TextBlock> getData() {
        return this.data;
    }

    // Returns the List of labels associated with the data of this DataLoader
    public List<String> getLabels() {
        return this.labels;
    }

    public static final Random RAND = new Random();

    // Shuffles the data and labels stored by the provided DataLoader equally so each label
    //      still corresponds to the original datapoint at the same index
    // 'loader' should be non-null.
    public static void shuffle(DataLoader loader) {
        DataLoader.shuffle(loader.data, loader.labels);
    }

    // Shuffles the provided data and label lists equally so each label still corresponds
    //      to the original datapoint at the same index
    // 'data' and 'labels' should be non-null.
    public static void shuffle(List<TextBlock> data, List<String> labels) {
        int seed = RAND.nextInt(Integer.MAX_VALUE);
        Collections.shuffle(data, new Random(seed));
        Collections.shuffle(labels, new Random(seed));
    }
}
