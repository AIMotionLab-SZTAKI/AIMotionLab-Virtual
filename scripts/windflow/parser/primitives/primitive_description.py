class PrimitiveDescription:
    def __init__(self, name, type, rgba, position, rotation):
        self.name = name
        self.type = type
        self.rgba = rgba
        self.position = position
        self.rotation = rotation

    def print(self):
        print(f'name: {self.name}')
        print(f'type: {self.type}')
        print(f'rgba: {self.rgba}')
        print(f'position: {self.position}')
        print(f'rotation: {self.rotation}')
