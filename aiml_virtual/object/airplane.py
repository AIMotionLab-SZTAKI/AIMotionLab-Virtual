from aiml_virtual.object.moving_object import MovingObject

class Airplane(MovingObject):

    def __init__(self, model, data, name_in_xml) -> None:
        super().__init__(model, name_in_xml)

        self.data = data

        self.qpos = data.joint(name_in_xml).qpos
    

    def get_qpos(self):

        return self.qpos
    
    def update(self, i, control_step):
        pass