from .language import PROFFESSIONS_EN, FANTASY_EN, RACE_EN, HUMANS_RU, MIMICS_VERBS_RU, WEAR_VERBS_RU, COLORS_RU
from .tags import SUBJECTS, ADJECTIVES, MORE_VERBS


def get_relatives(doc, obj):
    for rel in doc.tokens:
        if rel.head_id == obj.id:
            yield rel


def is_negated(doc, obj):
    negated = False
    for rel in get_relatives(doc, obj):
        if rel.lemma == 'не':
            negated = True
            break
    return negated


def get_prep_and_adjs(doc, root):
    res = []
    for token in doc.tokens:
        if token.head_id == root.id and token.rel == 'case':
            res.append(token)
    for token in doc.tokens:
        if token.head_id == root.id and token.rel == 'amod':
            res.append(token)
    return res


def translate(translator, text):
    return translator.translate(text, src='ru').text.lower()


def is_gd_feature(token):
    return False
    # return token.lemma in COLORS_RU


def get_objects(doc, translator):
    objects = []
    for token in doc.tokens:
        if (token.rel in SUBJECTS
                and (token.pos == 'PRON'
                     or token.lemma in HUMANS_RU
                     or translator.translate(token.lemma, src='ru').text.lower()
                     in PROFFESSIONS_EN
                     or translator.translate(token.lemma, src='ru').text.lower()
                     in RACE_EN
                     or translator.translate(token.lemma, src='ru').text.lower()
                     in FANTASY_EN)):
            objects.append(token)
    for span in doc.spans:
        if span.type == 'PER':
            for token in span.tokens:
                if token.rel in SUBJECTS and token not in objects:
                    objects.append(token)
    for obj in objects:
        yield is_negated(doc, obj), obj, is_gd_feature(obj)


def get_descriptions(doc, root):
    root_rels = get_relatives(doc, root)

    # get direct adj descriptions
    for token in root_rels:
        if token.pos == 'ADJ':
            yield ('не ' if is_negated(doc, token) else '') + token.text, is_gd_feature(token)

    relatives = [t for t in doc.tokens
                 if t.head_id == root.id
                 and t.id != root.id
                 and t.rel in ADJECTIVES
                 and t.pos != 'ADJ']
    for relative in relatives:
        if relative.pos == 'PART':
            continue
        rel_desc = []
        is_gd = is_gd_feature(relative)
        for token in doc.tokens:
            if token.head_id == relative.id:
                if token.rel == 'case' or token.rel == 'amod':
                    rel_desc.append(token.text)
                    is_gd = is_gd or is_gd_feature(token)
        yield " ".join(rel_desc) + " " + relative.text, is_gd

    # get mimics verbs
    for verb in doc.tokens:
        if (verb.id != root.head_id
            or verb.pos != 'VERB'
            or is_negated(doc, verb))\
                and not (verb.rel in MORE_VERBS
                         and verb.head_id == root.head_id):
            continue
        is_verb_gd = is_gd_feature(verb)
        if verb.lemma in MIMICS_VERBS_RU:
            yield verb.text, is_verb_gd
            continue
        elif verb.lemma in WEAR_VERBS_RU:
            is_gd = is_verb_gd
            verb_desc = []
            for verb_rel in get_relatives(doc, verb):
                if verb_rel.pos == 'NOUN' or verb_rel.rel == 'case' or verb_rel.pos == 'ADJ':
                    if verb_rel.id != root.id:
                        for tok in get_prep_and_adjs(doc, verb_rel):
                            verb_desc.append(tok.text)
                            is_gd = is_gd or is_gd_feature(tok)
                        verb_desc.append(verb_rel.text)
                        is_gd = is_gd or is_gd_feature(verb_rel)
            yield verb.text + " " + " ".join(verb_desc), is_gd
        else:
            is_gd = False
            for verb_rel in get_relatives(doc, verb):
                if verb_rel.rel == 'obl' and verb_rel.id != root.id:
                    verb_desc = []
                    for tok in get_prep_and_adjs(doc, verb_rel):
                        verb_desc.append(tok.text)
                        is_gd = is_gd or is_gd_feature(tok)
                    verb_desc.append(verb_rel.text)
                    is_gd = is_gd or is_gd_feature(verb_rel)
                    yield " ".join(verb_desc), is_gd
