package de.tudarmstadt.ukp.gleipnir;

import static org.apache.commons.io.IOUtils.closeQuietly;
import static org.apache.uima.fit.util.JCasUtil.indexCovering;
import static org.apache.uima.fit.util.JCasUtil.select;
import static org.apache.uima.fit.util.JCasUtil.selectCovered;

import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.core.api.io.JCasFileWriter_ImplBase;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.parameter.ComponentParameters;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;

public class ConnelWriter
        extends JCasFileWriter_ImplBase
{

    public static final String PARAM_TARGET_ENCODING = ComponentParameters.PARAM_TARGET_ENCODING;
    @ConfigurationParameter(name = PARAM_TARGET_ENCODING, mandatory = true,
            defaultValue = ComponentParameters.DEFAULT_ENCODING)
    private String targetEncoding;

    public static final String PARAM_FILENAME_EXTENSION =
            ComponentParameters.PARAM_FILENAME_EXTENSION;
    @ConfigurationParameter(name = PARAM_FILENAME_EXTENSION, mandatory = true, defaultValue = ".conll")
    private String filenameSuffix;

    private int autoDocumentId = 0;

    @Override
    public void process(JCas aJCas) throws AnalysisEngineProcessException {
        PrintWriter out = null;
        try {
            out = new PrintWriter(new OutputStreamWriter(getOutputStream(aJCas, filenameSuffix),
                    targetEncoding));

            String documentId = DocumentMetaData.get(aJCas).getDocumentId();

            out.printf("-DOCSTART- (%d %s)", autoDocumentId, documentId);

            convert(aJCas, out);
            out.println();

            autoDocumentId += 1;
        }
        catch (Exception e) {
            throw new AnalysisEngineProcessException(e);
        }
        finally {
            closeQuietly(out);
        }
    }

    private void convert(JCas aJCas, PrintWriter aOut) {
        Map<Token, Collection<NamedEntity>> neIdx = indexCovering(aJCas, Token.class,
                NamedEntity.class);

        Set<NamedEntity> processedNe = new HashSet<>();

        for (Sentence sentence : select(aJCas, Sentence.class)) {
            List<Token> tokens = selectCovered(Token.class, sentence);

            for (int i = 0; i < tokens.size(); i++) {
                Token token = tokens.get(i);

                aOut.println();
                aOut.printf("%s", token.getCoveredText());

                // If there are multiple named entities for the current token, we keep only the
                // first
                Collection<NamedEntity> neForToken = neIdx.get(token);
                if (neForToken == null || neForToken.isEmpty()) {
                    continue;
                }

                NamedEntity ne = neForToken.iterator().next();

                if(processedNe.contains(ne)) {
                    continue;
                } else {
                    processedNe.add(ne);
                }

                int neLength = JCasUtil.selectCovered(Token.class, ne).size();

                aOut.printf("\t%d:%d\t", i, i + neLength);

                if (ne.getIdentifier() == null) {
                    aOut.printf("NIL");
                } else {
                    aOut.printf("%s", ne.getIdentifier());
                }

            }

            aOut.println();
        }


        /*
        EU	B	EU	--NME--
        rejects
        German	B	German	Germany	http://en.wikipedia.org/wiki/Germany	11867	/m/0345h
        call
                to
        boycott
        British	B	British	United_Kingdom	http://en.wikipedia.org/wiki/United_Kingdom	31717	/m/07ssc
        lamb
                .
         */
    }
}
