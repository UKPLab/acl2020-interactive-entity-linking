/*
 * Copyright 2017
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.dkpro.core.vendor.io.tei;

import static org.dkpro.core.vendor.io.tei.internal.TeiConstants.TAG_PERS_NAME;
import static org.dkpro.core.vendor.io.tei.internal.TeiConstants.TAG_PLACE_NAME;
import static org.dkpro.core.vendor.io.tei.internal.TeiConstants.TAG_TEXT;
import static java.util.Arrays.asList;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.apache.uima.UimaContext;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.descriptor.MimeTypeCapability;
import org.apache.uima.fit.descriptor.ResourceMetaData;
import org.apache.uima.fit.descriptor.TypeCapability;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Logger;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import de.tudarmstadt.ukp.dkpro.core.api.io.ResourceCollectionReaderBase;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.parameter.MimeTypes;
import eu.openminted.share.annotations.api.DocumentationResource;

/**
 * Reader for the TEI XML.
 */
@ResourceMetaData(name = "TEI XML Reader")
@DocumentationResource("${docbase}/format-reference.html#format-${command}")
@MimeTypeCapability({MimeTypes.APPLICATION_TEI_XML})
@TypeCapability(
        outputs = {
                "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData",
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph",
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
                "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
                "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.constituent.Constituent",
                "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"})
public class TeiReader
        extends ResourceCollectionReaderBase
{
    private Resource currentResource;

    @Override
    public void initialize(UimaContext aContext)
            throws ResourceInitializationException
    {
        super.initialize(aContext);

        SAXParserFactory saxParserFactory = SAXParserFactory.newInstance();
    }

    @Override
    public void getNext(CAS aCAS)
            throws IOException, CollectionException
    {
        currentResource = nextFile();

        initCas(aCAS, currentResource);

        try(InputStream is = currentResource.getInputStream()) {
            JCas jcas = aCAS.getJCas();

            // Create handler
            Handler handler = new TeiHandler();
            handler.setJCas(jcas);
            handler.setLogger(getLogger());

            // Parse TEI text
            SAXParserFactory factory = SAXParserFactory.newInstance();
            SAXParser saxParser = factory.newSAXParser();

            saxParser.parse(is, handler);
        }
        catch (Exception e) {
            throw new CollectionException(e);
        }
    }

    protected abstract static class Handler
            extends DefaultHandler
    {
        private JCas jcas;
        private Logger logger;

        public void setJCas(final JCas aJCas)
        {
            jcas = aJCas;
        }

        protected JCas getJCas()
        {
            return jcas;
        }

        public void setLogger(Logger aLogger)
        {
            logger = aLogger;
        }

        public Logger getLogger()
        {
            return logger;
        }
    }

    public class TeiHandler
            extends Handler
    {
        private String documentId = null;
        private boolean titleSet = false;
        private boolean inTextElement = false;
        private boolean captureText = false;
        private boolean skipWhiteSpace = false;
        private Stack<NamedEntity> namedEntities = new Stack<>();

        private final StringBuilder buffer = new StringBuilder();

        private Set<String> capturedTags;

        private TeiHandler() {
            capturedTags = new HashSet<>();
            List<String> tags = asList(
                "p"
            );

            capturedTags.addAll(tags);
        }

        @Override
        public void endDocument()
        {
            getJCas().setDocumentText(buffer.toString());
        }

        @Override
        public void startElement(String aUri, String aLocalName, String aName,
                                 Attributes aAttributes)
        {
            if (inTextElement && capturedTags.contains(aName)) {
                captureText = true;
            }

            if (TAG_TEXT.equals(aName)) {
                inTextElement = true;
            }
            else if (inTextElement && TAG_PERS_NAME.equals(aName)) {
                NamedEntity ne = new NamedEntity(getJCas());
                ne.setBegin(buffer.length());
                ne.setValue("PER");

                if (aAttributes.getValue("ref") != null) {
                    String ref = aAttributes.getValue("ref");
                    ref = ref.replaceAll("^p:", "");
                    // Sometimes (very rarely), the ref contains two entities, we choose the first
                    ref = ref.split(" ")[0];
                    String iri = String.format("http://www.wwp.brown.edu/ns/1.0#%s", ref);
                    ne.setIdentifier(iri);
                }

                namedEntities.push(ne);
            }
            else if (inTextElement && TAG_PLACE_NAME.equals(aName)) {
                NamedEntity ne = new NamedEntity(getJCas());
                ne.setBegin(buffer.length());
                ne.setValue("LOC");

                namedEntities.push(ne);
            }  else if (inTextElement && ("hi".equals(aName))) {
                if (!skipWhiteSpace) {
                    buffer.append(" ");
                }
            }

            // System.out.printf("%b START Capture: %s %n", captureText, aLocalName);
        }

        @Override
        public void endElement(String aUri, String aLocalName, String aName)
                throws SAXException
        {
            if (TAG_TEXT.equals(aName)) {
                captureText = false;
                inTextElement = false;
            }
            else if (inTextElement && TAG_PERS_NAME.equals(aName)) {
                NamedEntity ne = namedEntities.pop();
                ne.setEnd(buffer.length());
                ne.addToIndexes();
            } else if (inTextElement && ("lb".equals(aName) || "l".equals(aName) || "p".equals(aName))) {
                if (!skipWhiteSpace) {
                    buffer.append("\n");
                }
            }  else if (inTextElement && ("hi".equals(aName))) {
                if (!skipWhiteSpace) {
                    buffer.append(" ");
                }
            }

            // System.out.printf("%b END Capture: %s %n", captureText, aLocalName);
        }

        @Override
        public void characters(char[] aCh, int aStart, int aLength)
                throws SAXException
        {
            if (captureText) {
                String strValue = new String(aCh, aStart, aLength);
                
                strValue = strValue.replace('ſ', 's');
                
                if (skipWhiteSpace) {
                    int firstNonWhitespace = 0;
                    boolean seenNonWhitespace = false;
                    while (firstNonWhitespace < strValue.length()) {
                        if (!Character.isWhitespace(strValue.charAt(firstNonWhitespace))) {
                            seenNonWhitespace = true;
                            break;
                        }
                        firstNonWhitespace ++;
                    }
                 
                    // If there was any non-whitespace, we add it to the buffer and return to the
                    // normal capture mode
                    if (seenNonWhitespace) {
                        strValue = strValue.substring(firstNonWhitespace, strValue.length());
                        skipWhiteSpace = false;
                    }
                    // ... if there was only whitespace, we skip the whole thing
                    else {
                        return;
                    }
                }
                
                int softHyphenIndex = strValue.indexOf(0x00AD);
                if (softHyphenIndex > -1) {
                    strValue = strValue.substring(0, softHyphenIndex);
                    skipWhiteSpace = true;
                }
                
                
                buffer.append(strValue);
            }
        }
   }
}