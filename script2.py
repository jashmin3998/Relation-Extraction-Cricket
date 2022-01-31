import spacy
import yaml

from rel_component.scripts.linker_utils import attach_linker
from rel_component.scripts.rel_pipe import make_relation_extractor, score_relations
from rel_component.scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# helper functions and classes
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    _getattr_ = dict.get
    _setattr_ = dict._setitem_
    _delattr_ = dict._delitem_


class PipelineModel:
    def _init_(self) -> None:
        self.__pipeline_config = self.load_yaml_file('./pipeline.yml')
        self._ner = spacy.load(self._pipeline_config.vars['ner_model'])
        self._ner = attach_linker(spacy_model=self._ner)
        self._re = spacy.load(self._pipeline_config.vars['re_model'])

    @staticmethod
    def load_yaml_file(file_name):
        with open(file_name) as file:
            doc_dict = yaml.full_load(file)
        return DotDict(doc_dict)

    def get_predictions(self, text: str, threshold: float = 0.4):
        doc = self.__ner(text)
        print(f"Text: {text}\n")
        print(f"Extracted Entities: {[(e.text, e.label_) for e in doc.ents]}\n")
        linker = self.__ner.get_pipe("scispacy_linker")
        for entity in doc.ents:
            for ent in entity._.kb_ents:
                print(linker.kb.cui_to_entity[ent[0]])

        for name, proc in self.__re.pipeline:
            doc = proc(doc)

        for value, rel_dict in doc._.rel.items():
            for e in doc.ents:
                for b in doc.ents:
                    if e.start == value[0] and b.start == value[1]:
                        if rel_dict['Binds'] >= threshold:
                            print(
                                f"Entities: {e.text, b.text} --> Predicted Relation: Binds")
                        elif rel_dict['Regulates'] >= threshold:
                            print(
                                f"Entities: {e.text, b.text} --> Predicted Relation: Regulates")


if __name__ == "_main_":
    pipeline = PipelineModel()
    sample_text = "The up-regulation of Id1 mRNA was characteristic of an early inducible gene, with " \
        "maximal upregulation two hours after the addition of BMP-6 and returned to baseline after 24 hours."
    pipeline.get_predictions(text=sample_text)