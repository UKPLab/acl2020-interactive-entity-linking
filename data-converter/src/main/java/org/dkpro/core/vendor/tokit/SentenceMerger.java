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

package org.dkpro.core.vendor.tokit;

import static org.apache.uima.fit.util.CasUtil.select;
import static org.apache.uima.fit.util.JCasUtil.selectCovered;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.apache.uima.UimaContext;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.Feature;
import org.apache.uima.cas.Type;
import org.apache.uima.cas.text.AnnotationFS;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.descriptor.ResourceMetaData;
import org.apache.uima.fit.descriptor.TypeCapability;
import org.apache.uima.fit.util.CasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import eu.openminted.share.annotations.api.DocumentationResource;

/**
 * Merges any Tokens that are covered by a given annotation type. E.g. this component can be used
 * to create a single tokens from all tokens that constitute a multi-token named entity.
 */
@ResourceMetaData(name = "Token Merger")
@DocumentationResource("${docbase}/component-reference.html#engine-${shortClassName}")
@TypeCapability(
        inputs = { "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"},
        outputs = {
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"})
public class SentenceMerger
    extends JCasAnnotator_ImplBase
{
    /**
     * Annotation type for which sentences should be merged.
     */
    public static final String PARAM_ANNOTATION_TYPE = "annotationType";
    @ConfigurationParameter(name = PARAM_ANNOTATION_TYPE, mandatory = true)
    private String annotationType;

    @Override
    public void initialize(UimaContext aContext) throws ResourceInitializationException
    {
        super.initialize(aContext);
    }

    @Override
    public void process(JCas aJCas)
    {
        CAS cas = aJCas.getCas();

        List<AnnotationFS> covers = new ArrayList<>(
                select(cas, CasUtil.getAnnotationType(cas, annotationType)));
        Collection<AnnotationFS> toRemove = new ArrayList<>();

        Type sentenceType = aJCas.getCas().getTypeSystem().getType("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence");
        Feature endFeature = aJCas.getCas().getEndFeature();

        for (AnnotationFS cover : covers) {

            List<AnnotationFS> covered = selectOverlapping(aJCas.getCas(), sentenceType, cover.getBegin(), cover.getEnd());
            if (covered.size() < 2) {
                continue;
            }

            Iterator<AnnotationFS> i = covered.iterator();

            // Extend first sentence
            AnnotationFS sentence = i.next();
            aJCas.removeFsFromIndexes(sentence);
            sentence.setIntValue(endFeature, covered.get(covered.size() - 1).getEnd());
            aJCas.addFsToIndexes(sentence);

            // Mark the rest for deletion
            while (i.hasNext()) {
                AnnotationFS s = i.next();
                toRemove.add(s);
            }
        }

        // Remove sentences no longer needed
        for (AnnotationFS t : toRemove) {
            aJCas.removeFsFromIndexes(t);
        }
    }

    public static List<AnnotationFS> selectOverlapping(CAS aCas,
                                                       Type aType, int aBegin, int aEnd)
    {

        List<AnnotationFS> annotations = new ArrayList<>();
        for (AnnotationFS t : select(aCas, aType)) {
            if (t.getBegin() >= aEnd) {
                break;
            }
            // not yet there
            if (t.getEnd() <= aBegin) {
                continue;
            }
            annotations.add(t);
        }

        return annotations;
    }
}