package de.tudarmstadt.ukp.gleipnir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Splitter {
    public static void doTheSplit(Path documentFolder, Path splitsPath, Path generatedPath,
                                  ThreeParameterFunction<Path, Path, String> converter) throws Exception {
        List<Path> files = Files.list(documentFolder).sorted().collect(Collectors.toList());

        int totalSize = files.size();
        int trainSetSize = (int) Math.floor((totalSize * 0.65));
        int testSetSize = (int) (Math.ceil(totalSize * 0.15));

        List<Path> train = new ArrayList<>();
        List<Path> dev = new ArrayList<>();
        List<Path> test = new ArrayList<>();

        for (int i = 0; i < trainSetSize; i++) {
            train.add(files.remove(0));
        }

        for (int i = 0; i < testSetSize; i++) {
            test.add(files.remove(0));
        }
        dev = files;

        System.out.printf("Train: %d, Dev: %d, Test: %d\n", train.size(), dev.size(), test.size());

        Path trainPath = splitsPath.resolve("train");
        Path devPath = splitsPath.resolve("dev");
        Path testPath = splitsPath.resolve("test");

        copyStuff(train, trainPath);
        copyStuff(dev, devPath);
        copyStuff(test, testPath);

        converter.apply(trainPath, generatedPath, "train");
        converter.apply(devPath, generatedPath, "dev");
        converter.apply(testPath, generatedPath, "test");
    }

    public static void copyStuff(List<Path> fileNames, Path targetFolder) throws Exception {
        if(Files.notExists(targetFolder)) {
            Files.createDirectories(targetFolder);
        }

        for (Path fileName : fileNames) {
            Path targetPath = targetFolder.resolve(fileName.getFileName());
            Files.copy(fileName, targetPath, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    @FunctionalInterface
    public interface ThreeParameterFunction<T, U, V> {
        public void apply(T t, U u, V v);
    }
}
