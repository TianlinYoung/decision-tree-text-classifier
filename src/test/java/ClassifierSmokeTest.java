import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class ClassifierSmokeTest {
    @TempDir
    Path temp;

    @Test
    public void testTrainAndPredict() {
        List<TextBlock> data = Arrays.asList(new TextBlock("apple apple"),
                                             new TextBlock("stone stone"));
        Classifier model = new Classifier(data, Arrays.asList("fruit", "other"));
        assertEquals("fruit", model.classify(new TextBlock("apple")));
        assertEquals("other", model.classify(new TextBlock("stone")));
    }

    @Test
    public void testSaveLoadRoundTrip() throws IOException {
        List<TextBlock> data = Arrays.asList(new TextBlock("apple apple"),
                                             new TextBlock("stone stone"));
        Classifier original = new Classifier(data, Arrays.asList("fruit", "other"));
        Path saved = temp.resolve("tree.txt");
        try (PrintStream output = new PrintStream(saved.toFile())) {
            original.save(output);
        }
        Classifier loaded;
        try (Scanner input = new Scanner(saved.toFile())) {
            loaded = new Classifier(input);
        }
        for (String text : Arrays.asList("apple", "stone", "unknown", "")) {
            assertEquals(original.classify(new TextBlock(text)),
                         loaded.classify(new TextBlock(text)));
        }
        Path savedAgain = temp.resolve("again.txt");
        try (PrintStream output = new PrintStream(savedAgain.toFile())) {
            loaded.save(output);
        }
        assertEquals(Files.readString(saved), Files.readString(savedAgain));
    }

    @Test
    public void testQuotedCommaAndEmptyMiddleField() throws IOException {
        Path csv = temp.resolve("quoted.csv");
        Files.writeString(csv, "Label,Message,Note\nfruit,\"apple, pear\",ok\nother,,empty\n");
        List<List<String>> rows = CsvReader.read(csv.toString());
        assertEquals(2, rows.size());
        // The existing parser retains quote characters, but does not split the quoted comma.
        assertEquals(Arrays.asList("fruit", "\"apple, pear\"", "ok"), rows.get(0));
        assertEquals(Arrays.asList("other", "", "empty"), rows.get(1));
    }

    @Test
    public void testCsvToModelPipeline() throws IOException {
        DataLoader loader = new DataLoader("examples/sample.csv", 0, 1, false);
        assertEquals(Arrays.asList("fruit", "other"), loader.getLabels());
        assertEquals(2, loader.getData().size());
        assertEquals(1.0, loader.getData().get(0).get("apple"));
        Classifier model = new Classifier(loader.getData(), loader.getLabels());
        assertEquals("fruit", model.classify(new TextBlock("apple")));
        assertEquals("other", model.classify(new TextBlock("stone")));
    }
}
