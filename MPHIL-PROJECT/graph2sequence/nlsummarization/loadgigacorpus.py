#!/usr/bin/env python
"""
Usage:
    loadgigacorpus.py [options] INPUT_XML_FOLDER OUT_FILEPATH_PREFIX

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import codecs
import gzip
import json
import os
import pdb
import sys
import traceback
from typing import Iterable, List, Optional, Tuple

from docopt import docopt
from nltk.tree import Tree

from graph2sequence.nlsummarization.graphtextrepr import (DependencyEdge,
                                                          GraphTextRepresentation,
                                                          Token)
from graph2sequence.nlsummarization.textsummary import TextSummary
from graph2sequence.nlsummarization.utils import load_xml_gz


def parse_tree_to_sentence(parse_tree:str)-> List[str]:
    return Tree.fromstring(parse_tree).leaves()

def try_find_RB_span(tokens: List[str]) -> Optional[Tuple[int, int]]:
    try:
        lrb_idx = tokens.index('-LRB-')
        rrb_idx = tokens.index('-RRB-')
        return (lrb_idx, rrb_idx+1)
    except ValueError:
        return None


def parse_sample(datapoint)-> Optional[TextSummary]:
    if datapoint.get('HEADLINE') is None or len(datapoint['HEADLINE']) == 0:
        print('No headline found. Ignoring sample.')
        return None
    try:
        headline_tokens = parse_tree_to_sentence(datapoint['HEADLINE'])
        # Remove LRB-RRB chunks
        rb_span = try_find_RB_span(headline_tokens)
        while rb_span is not None:
            headline_tokens = headline_tokens[:rb_span[0]] + headline_tokens[rb_span[1]:]
            rb_span = try_find_RB_span(headline_tokens)

    except Exception as e:
        print('Could not parse %s. Ignoring sample.' % datapoint['HEADLINE'])
        return None

    all_sentences = datapoint['sentences']['sentence']
    if type(all_sentences) is not list:
        all_sentences = [all_sentences]

    tokenized_sentences = []  # type: List[List[Token]]
    for sentence in all_sentences:
        sentence_tokens = []
        if type(sentence['tokens']['token']) is not list:
            # May happen in single-word sentences
            sentence['tokens']['token'] = [sentence['tokens']['token']]
        for i, token in enumerate(sentence['tokens']['token']):
            assert int(token['@id']) == i + 1
            sentence_tokens.append(Token(word=token['word'], lemma=token['lemma'], pos_tag=token['POS']))
        tokenized_sentences.append(sentence_tokens)

    graph_text_representation = GraphTextRepresentation(tokenized_sentences)
    
    # Add named entities, by finding consecutive annotations
    for sentence_idx, sentence in enumerate(all_sentences):
        sentence_tokens = sentence['tokens']['token']
        for token_idx, token in enumerate(sentence_tokens):
            if token['NER'] == 'O':
                continue
            if token_idx + 1 < len(sentence_tokens) - 1 and sentence_tokens[token_idx + 1]['NER'] != token['NER']:
                # Create an entity that includes this token as the last one
                before_start_token_idx = token_idx - 1
                while before_start_token_idx > 0 and sentence_tokens[before_start_token_idx]['NER'] == token['NER']:
                    before_start_token_idx -= 1
                graph_text_representation.add_entity(token['NER'], sentence_idx, before_start_token_idx + 1, token_idx + 1)

    # Add dependencies
    for sentence_idx, sentence in enumerate(all_sentences):
        if sentence['collapsed-dependencies'] is None:
            continue
        if type(sentence['collapsed-dependencies']['dep']) is not list:
            sentence['collapsed-dependencies']['dep'] = [sentence['collapsed-dependencies']['dep']]
        for dependency in sentence['collapsed-dependencies']['dep']:
            if dependency['@type'] == 'root':
                continue  # Root is not useful for us
            graph_text_representation.add_dependency_edge(DependencyEdge(
                dependency_type=dependency['@type'],
                sentence_idx=sentence_idx,
                from_idx=int(dependency['dependent']) - 1,
                to_idx=int(dependency['governor']) - 1
            ))

    # Add co-references
    if datapoint['coreferences'] is not None:
        if type(datapoint['coreferences']['coreference']) is not list:
            datapoint['coreferences']['coreference'] = [datapoint['coreferences']['coreference']]
        for coreference in datapoint['coreferences']['coreference']:
            all_mentions = coreference['mention']
            representative = [m for m in all_mentions if '@representative' in m and m['@representative'] == 'true'][0]

            for mention in all_mentions:
                if mention.get('@representative') == 'true' or (mention['sentence'] == representative['sentence'] and mention['head'] == representative['head']):
                    continue
                graph_text_representation.add_coreference(int(mention['sentence']) - 1, int(mention['head']) - 1,
                                                          int(representative['sentence']) -1, int(representative['head'])-1)

    return TextSummary(
        summary_sentence=headline_tokens,
        main_text= graph_text_representation
    )

def parse_gigacorpus_file(filename: str) -> Iterable[TextSummary]:
    data = load_xml_gz(filename)
    for doc in data['FILE']['DOC']:
        sample = parse_sample(doc)
        if sample is not None:
            yield sample

def run(args):
    with gzip.GzipFile(args['OUT_FILEPATH_PREFIX'] + '-inputs.jsonl.gz', 'wb') as text_input_file:
        with  gzip.GzipFile(args['OUT_FILEPATH_PREFIX'] + '-summaries.jsonl.gz', 'wb') as summaries_file:
            num_points = 0
            writer = codecs.getwriter('utf-8')
            for file in os.listdir(args['INPUT_XML_FOLDER']):
                if not file.endswith('.xml.gz'): continue
                full_file_path = os.path.join(args['INPUT_XML_FOLDER'], file)
                print('Loading %s...' % full_file_path)

                for textsum in parse_gigacorpus_file(full_file_path):
                    if len(textsum.summary_sentence) <= 1:
                        print('Single word headline: %s. Ignoring...' % textsum.summary_sentence)
                        continue
                    json.dump(textsum.main_text.to_graph_object(), writer(text_input_file))
                    writer(text_input_file).write('\n')

                    json.dump(textsum.summary_sentence, writer(summaries_file))
                    writer(summaries_file).write('\n')
                    num_points += 1
    print('Loaded %s datapoints' % num_points)

if __name__ == '__main__':
    args = docopt(__doc__)
    try:
        run(args)
    except:
        if args.get('--debug', False):
            _, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise
