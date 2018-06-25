import json
from typing import List, NamedTuple

from graph2sequence.nlsummarization.graphtextrepr import GraphTextRepresentation

TextSummary = NamedTuple('TextSummary', [('summary_sentence', List[str]), ('main_text', GraphTextRepresentation)])

def summary_to_json_string(summary: TextSummary) -> str:
    graph = summary.main_text.to_graph_object()
    return json.dumps({'graph': graph, 'target_sequence': summary.summary_sentence})


    