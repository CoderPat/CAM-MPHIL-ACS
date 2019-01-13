from typing import List, NamedTuple

from graph2sequence.nlsummarization.graphtextrepr import GraphTextRepresentation

TextSummary = NamedTuple('TextSummary', [('summary_sentence', List[str]), ('main_text', GraphTextRepresentation)])



    