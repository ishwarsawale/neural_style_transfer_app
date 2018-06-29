#!/usr/bin python3
from kivy.app import App
from imagemix import imagemixController

class imagemixApp(App):
    def build(self):
        self.title = 'Neural_Style_App'
        return imagemixController()


if __name__ == "__main__":
    imagemixApp().run()
