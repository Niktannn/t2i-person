from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from googletrans import Translator

from .utils import get_objects, get_descriptions, translate


def process_text(text):
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    translator = Translator()

    doc = Doc(text)

    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    ans = []

    for (negated, token, is_gd_obj) in get_objects(doc, translator):
        descriptions = [(translate(translator, desc), is_gd_desc) for desc, is_gd_desc in get_descriptions(doc, token)]
        ans.append({'obj': (("not " if negated else "") + translate(translator, token.lemma), is_gd_obj),
                    'desc': descriptions,
                    'gen': token.feats.get('Gender')})

    return ans
