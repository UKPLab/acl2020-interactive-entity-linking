package de.tudarmstadt.ukp.gleipnir.wwo;

import static de.tudarmstadt.ukp.gleipnir.Splitter.doTheSplit;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;
import static org.apache.uima.fit.pipeline.SimplePipeline.runPipeline;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.uima.fit.factory.AnalysisEngineFactory;

import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.io.xmi.XmiWriter;
import de.tudarmstadt.ukp.dkpro.core.jtok.JTokSegmenter;
import de.tudarmstadt.ukp.dkpro.core.tokit.BreakIteratorSegmenter;


import org.dkpro.core.vendor.io.tei.TeiReader;
import org.dkpro.core.vendor.nif.NifReader;
import org.dkpro.core.vendor.nif.NifWriter;
import org.dkpro.core.vendor.tokit.SentenceMerger;
import de.tudarmstadt.ukp.gleipnir.ConnelWriter;

/**
 * Hello world!
 *
 */
public class AppWwo
{
    public static void main( String[] args ) throws Exception
    {
        convertWwo();
    }

    public static void convertWwo() throws Exception {
        Path path = Paths.get("/home/klie/git/linker/data/wwo/files/");
        Path generatedPath = Paths.get("/home/klie/git/linker/generated/wwo/");
        Path splitsPath = generatedPath.resolve("splits");

        doTheSplit(path, splitsPath, generatedPath, (a,b,c) -> {
            try { convertSingleWwo(a,b,c); } catch (Exception e) { throw new RuntimeException(e); }
        });
    }

    public static void convertSingleWwo(Path pathIn, Path pathOut, String name) throws Exception {
        Path pathOutXmi = pathOut.resolve("xmi").resolve(name);
        if(Files.notExists(pathOutXmi)) {
            Files.createDirectories(pathOutXmi);
        }

        String pathOutConnel = pathOut.resolve(name + ".conll").toString();

        runPipeline(
                createReaderDescription(TeiReader.class,
                        TeiReader.PARAM_SOURCE_LOCATION, pathIn.resolve("*.xml").toString(),
                        TeiReader.PARAM_LANGUAGE, "en"),
                createEngineDescription(BreakIteratorSegmenter.class),
                createEngineDescription(SentenceMerger.class,
                        SentenceMerger.PARAM_ANNOTATION_TYPE, NamedEntity.class),
                createEngineDescription(XmiWriter.class,
                        XmiWriter.PARAM_TARGET_LOCATION, pathOutXmi.toString(),
                        XmiWriter.PARAM_OVERWRITE, true,
                        XmiWriter.PARAM_STRIP_EXTENSION, true,
                        XmiWriter.PARAM_TYPE_SYSTEM_FILE, "/dev/null"),
                AnalysisEngineFactory.createEngineDescription(ConnelWriter.class,
                        ConnelWriter.PARAM_TARGET_LOCATION, pathOutConnel.toString(),
                        ConnelWriter.PARAM_OVERWRITE, true,
                        ConnelWriter.PARAM_STRIP_EXTENSION, true,
                        ConnelWriter.PARAM_SINGULAR_TARGET, true));
    }

}
