class ColorText():
    """
    Use ANSI escape sequences to print colors +/- bold/underline to bash terminal.
    
    Notes
    -----
    execute ColorText.demo() for a printout of colors.
    """
    def demo():
        """Prints examples of all colors in normal, bold, underline, bold+underline."""
        for color in dir(ColorText):
            if all([color.startswith('_') is False,
                   color not in ['bold', 'underline', 'demo'],
                   callable(getattr(ColorText, color))]):
                print(getattr(ColorText(color), color)(),'\t',                
                      getattr(ColorText(f'bold {color}').bold(), color)(),'\t',
                      getattr(ColorText(f'underline {color}').underline(), color)(),'\t',
                      getattr(ColorText(f'bold underline {color}').underline().bold(), color)())
        pass

    def __init__(self, text:str):
        self.text = text
        self.ending = '\033[0m'
        self.colors = []
        pass

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text

    def bold(self):
        self.text = '\033[1m' + self.text + self.ending
        return self

    def underline(self):
        self.text = '\033[4m' + self.text + self.ending
        return self

    def green(self):
        self.text = '\033[92m' + self.text + self.ending
        self.colors.append('green')
        return self

    def purple(self):
        self.text = '\033[95m' + self.text + self.ending
        self.colors.append('purple')
        return self

    def blue(self):
        self.text = '\033[94m' + self.text + self.ending
        self.colors.append('blue')
        return self

    def ltblue(self):
        self.text = '\033[34m' + self.text + self.ending
        self.colors.append('lightblue')
        return self

    def pink(self):
        self.text = '\033[35m' + self.text + self.ending
        self.colors.append('pink')
        return self
    
    def gray(self):
        self.text = '\033[30m' + self.text + self.ending
        self.colors.append('gray')
        return self

    def ltgray(self):
        self.text = '\033[37m' + self.text + self.ending
        self.colors.append('ltgray')
        return self

    def warn(self):
        self.text = '\033[93m' + self.text + self.ending
        self.colors.append('yellow')
        return self

    def fail(self):
        self.text = '\033[91m' + self.text + self.ending
        self.colors.append('red')
        return self

    def ltred(self):
        self.text = '\033[31m' + self.text + self.ending
        self.colors.append('lightred')
        return self

    def cyan(self):
        self.text = '\033[36m' + self.text + self.ending
        self.colors.append('cyan')
        return self

    pass