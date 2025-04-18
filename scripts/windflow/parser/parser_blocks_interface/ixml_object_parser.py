from abc import ABC, abstractmethod


class IXMLObjectParser:
    @abstractmethod
    def get_primitives(self, source_xml_path):
        pass
