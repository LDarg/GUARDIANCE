import os
import pygame

class ResourceManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = self.get_dir_name()
        self.base_dir = base_dir
        self.book_icon = None
        self.child_icon = None
        self.exclamation_mark_icon = None

    @staticmethod
    def get_dir_name():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics")

    def load_icons(self, size=(100,100)):
        self.book_icon = self.load_and_scale("book.png", size)
        self.child_icon = self.load_and_scale("child.png", size)
        self.exclamation_mark_icon = self.load_and_scale("exclamation_mark.png", size)

    def load_and_scale(self, filename, size):
        path = os.path.join(self.base_dir, filename)
        image = pygame.image.load(path)
        return pygame.transform.scale(image, size)

