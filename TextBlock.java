import java.util.*;

// This class represents a piece of text data that can be classified
public class TextBlock {
    private Map<String, Integer> wordCounts;
    private double totalWords;

    // Constructs a new TextBlock from the provided content String
    public TextBlock(String content) {
        this.wordCounts = new HashMap<>();
        parseContent(content);
    }

    // Helper method - parses the content from the provided content String,
    //      populating the word -> count map and counting the total words/tokens
    private void parseContent(String content) {
        Scanner sc = new Scanner(content);
        while (sc.hasNext()) {
            String word = sc.next();
            if (!wordCounts.containsKey(word)) {
                wordCounts.put(word, 0);
            }
            wordCounts.put(word, wordCounts.get(word) + 1);
            totalWords++;
        }
    }

    // Returns the word probability for the given word.
    // (number of times the word appeared / total number of all words)
    // Returns 0 if this TextBlock doesn't have any words.
    public double get(String word) {
        if (totalWords != 0 && wordCounts.containsKey(word)) {
            return wordCounts.get(word) / totalWords;
        }

        return 0;
    }
    
    // Returns a Set of all valid features for this TextBlock.
    public Set<String> getFeatures() { return wordCounts.keySet(); }

    // Returns true if TextBlock contains this feature. False otherwise.
    public boolean containsFeature(String word) { return wordCounts.containsKey(word); }
    
    // Returns a feature that has the greatest difference in word probability between this
    // instance and provided 'other'
    public String findBiggestDifference(TextBlock other) {
        // Find the word with the largest probability difference between this and other
        Set<String> allWords = new HashSet<>(this.wordCounts.keySet());
        allWords.addAll(other.wordCounts.keySet());

        String bestWord = null;
        double highestDiff = 0;
        for (String word : allWords) {
            double diff = (this.containsFeature(word) ? this.get(word) : 0) - 
                          (other.containsFeature(word) ? other.get(word) : 0);
            diff = Math.abs(diff);
            if (diff > highestDiff) {
                bestWord = word;
                highestDiff = diff;
            }
        }

        return bestWord;
    }
}
