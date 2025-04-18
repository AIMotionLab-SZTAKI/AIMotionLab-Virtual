from abc import ABC, abstractmethod


class IPreprocessor:
    @abstractmethod
    def preprocess_xml(self, source_xml_path, final_xml_path):
        pass
