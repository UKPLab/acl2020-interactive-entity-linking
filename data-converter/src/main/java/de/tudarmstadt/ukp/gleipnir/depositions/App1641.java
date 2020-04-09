package de.tudarmstadt.ukp.gleipnir.depositions;

import static de.tudarmstadt.ukp.gleipnir.Splitter.doTheSplit;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;
import static org.apache.uima.fit.pipeline.SimplePipeline.runPipeline;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;


import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.io.text.TextWriter;
import de.tudarmstadt.ukp.dkpro.core.io.xmi.XmiWriter;
import de.tudarmstadt.ukp.dkpro.core.jtok.JTokSegmenter;

import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.dkpro.core.vendor.nif.NifReader;
import org.dkpro.core.vendor.nif.NifWriter;
import org.dkpro.core.vendor.tokit.SentenceMerger;
import de.tudarmstadt.ukp.gleipnir.ConnelWriter;
import org.dkpro.core.vendor.io.tei.TeiReader;

public class App1641 {

    public static void main(String[] args) throws Exception {
        Path generatedPath = Paths.get("../linker/generated/depositions/");

        if(Files.notExists(generatedPath)) {
            Files.createDirectories(generatedPath);
        }

        convertDepositions();

//        runPipeline(
//                createReaderDescription(NifReader.class,
//                        NifReader.PARAM_SOURCE_LOCATION, "hist1641.ttl",
//                        NifReader.PARAM_IRI_PREFIX, "http://1641.tcd.ie/",
//                        TeiReader.PARAM_LANGUAGE, "en"),
//                createEngineDescription(JTokSegmenter.class),
//                createEngineDescription(SentenceMerger.class,
//                        SentenceMerger.PARAM_ANNOTATION_TYPE, NamedEntity.class),
//                createEngineDescription(ConnelWriter.class,
//                        ConnelWriter.PARAM_TARGET_LOCATION, "depositions",
//                        ConnelWriter.PARAM_OVERWRITE, true,
//                        ConnelWriter.PARAM_STRIP_EXTENSION, true,
//                        ConnelWriter.PARAM_SINGULAR_TARGET, false),
//                createEngineDescription(TextWriter.class,
//                        ConnelWriter.PARAM_TARGET_LOCATION, "depositions/txt",
//                        ConnelWriter.PARAM_OVERWRITE, true,
//                        ConnelWriter.PARAM_STRIP_EXTENSION, true,
//                        ConnelWriter.PARAM_SINGULAR_TARGET, false),
//                createEngineDescription(ConnelWriter.class,
//                        ConnelWriter.PARAM_TARGET_LOCATION, generatedPath.toString() + "train.conll",
//                        ConnelWriter.PARAM_OVERWRITE, true,
//                        ConnelWriter.PARAM_STRIP_EXTENSION, true,
//                        ConnelWriter.PARAM_SINGULAR_TARGET, true));
    }

    public static void convertDepositions() throws Exception {
        Path path = Paths.get("depositions");
        Path generatedPath = Paths.get("../linker/generated/depositions/");
        Path splitsPath = generatedPath.resolve("splits");

        deleteDirectoryStream(generatedPath);

        if(Files.notExists(generatedPath)) {
            Files.createDirectories(generatedPath);
        }

        // Split the nif into parts
        runPipeline(
                createReaderDescription(NifReader.class,
                        NifReader.PARAM_SOURCE_LOCATION, "hist1641.ttl",
                        NifReader.PARAM_IRI_PREFIX, "http://1641.tcd.ie/",
                        NifReader.PARAM_LANGUAGE, "en"),
                createEngineDescription(NifWriter.class,
                        NifWriter.PARAM_TARGET_LOCATION, path.toString(),
                        NifWriter.PARAM_OVERWRITE, true,
                        NifWriter.PARAM_SINGULAR_TARGET, false));

        doTheSplit(path, splitsPath, generatedPath, (a,b,c) -> {
            try { convertSingleDeposition(a,b,c); } catch (Exception e) { throw new RuntimeException(e); }
        });
    }

    public static void convertSingleDeposition(Path pathIn, Path pathOut, String name) throws Exception {
        Path pathOutXmi = pathOut.resolve("xmi").resolve(name);
        if(Files.notExists(pathOutXmi)) {
            Files.createDirectories(pathOutXmi);
        }

        String pathOutConnel = pathOut.resolve(name + ".conll").toString();

        runPipeline(
                createReaderDescription(NifReader.class,
                        NifReader.PARAM_SOURCE_LOCATION, pathIn.toString() + "/*.ttl",
                        NifReader.PARAM_IRI_PREFIX, "http://1641.tcd.ie/",
                        NifReader.PARAM_LANGUAGE, "en"),
                createEngineDescription(JTokSegmenter.class),
                createEngineDescription(SentenceMerger.class,
                        SentenceMerger.PARAM_ANNOTATION_TYPE, NamedEntity.class),
                createEngineDescription(XmiWriter.class,
                        XmiWriter.PARAM_TARGET_LOCATION, pathOutXmi.toString(),
                        XmiWriter.PARAM_OVERWRITE, true,
                        XmiWriter.PARAM_STRIP_EXTENSION, true,
                        XmiWriter.PARAM_TYPE_SYSTEM_FILE, "/dev/null"),
                AnalysisEngineFactory.createEngineDescription(ConnelWriter.class,
                        ConnelWriter.PARAM_TARGET_LOCATION, pathOutConnel,
                        ConnelWriter.PARAM_OVERWRITE, true,
                        ConnelWriter.PARAM_STRIP_EXTENSION, true,
                        ConnelWriter.PARAM_SINGULAR_TARGET, true));
    }

    static void deleteDirectoryStream(Path path) throws IOException {
        Files.walk(path)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
    }

}
