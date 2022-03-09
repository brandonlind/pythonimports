"""Personalized functions to interact with R.

TODO
----
- R.plot save/show pdf - right now it will save/show png etc but if pdf only can save but not show
"""
import os
from os import path as op
from contextlib import contextmanager
from IPython.display import Image, display, IFrame
from os import path as op
from pythonimports import ColorText
import tempfile


def _global_imports(Import: str, As: str = None, From: str = None):
    """Import from local function as global import.

    Use this statement to import inside a function,
    but effective as import at the top of the module.

    Parameters
    ----------
    Import: the object name want to import, could be module or function
    As: the short name for the import
    From: the context module name in the import

    Examples
    --------
    import os -> _global_imports(Import="os")
    import numpy as np -> _global_imports(Import="numpy", As="np")
    from collections import Counter ->
        _global_imports(From="collections", Import="Counter")
    from google.cloud import storage ->
        _global_imports(From="google.cloud", Import="storage")
        
    Notes
    -----
    thanks https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function
    - though 'import x as y' didn't always work for me eg 

    """
    if not As:
        As = Import
    if not From:
        globals()[As] = __import__(Import)
    else:
        context_module = __import__(From,
                                    fromlist=[Import])
        globals()[As] = getattr(context_module, Import)
        
    pass


def _setup_r(home=None, ld_library_path=''):
    """Set up R for python.
    
    Parameters
    ----------
    home: path to directory with `lib` subfolder - ie: assert(op.exists(op.join(home, 'lib')))
    ld_library_path: 
    """
    try:
        current_path = os.environ['LD_LIBRARY_PATH']
    except KeyError as e:
        current_path = ''
        
    if ld_library_path not in current_path:
        ld_library_path = f"{current_path}:{ld_library_path}"
    else:
        ld_library_path = current_path
            
    os.environ['R_HOME'] = home
    os.environ['LD_LIBRARY_PATH'] = ld_library_path
    
    _global_imports(Import='robjects', From='rpy2')
    _global_imports(Import='pandas2ri', From='rpy2.robjects')
    _global_imports(Import='grdevices', From='rpy2.robjects.lib')
    _global_imports(Import='importr', From='rpy2.robjects.packages')
    
    pandas2ri.activate()
    
#     if notebook is True:
    #     %load_ext rpy2.ipython - if I wanted to run cell magics
    
    r = robjects.r
    
    return r


@contextmanager
def _r_inline_plot(saveloc=None, _format='png', **kwargs):
    """Plot something from R inline in a jupyter notebook.
    
    Example
    -------
    with _r_inline_plot():
        graphics.plot(result)  # `result` is of type from rpy2.robjects (eg ListVector)

    Notes
    -----
    thanks - https://stackoverflow.com/questions/43110228/how-to-plot-inline-with-rpy2-in-jupyter-notebook
    """
#     print(f'{_format = }')  # for debugging
    
    #grdevices.__dict__[_format](file=saveloc, width=kwargs['width'], height=kwargs['height'])
    if _format != 'pdf':
        x = grdevices.render_to_bytesio(grdevices.__dict__[_format], 
                                        width=kwargs['width'],
                                        height=kwargs['height'],
                                        res=kwargs['dpi'])
    else:
        x = grdevices.render_to_bytesio(grdevices.__dict__[_format], 
                                     width=kwargs['width'],
                                     height=kwargs['height'])
    with x as b:
        yield

#     data = b.getvalue()
#     display(Image(data=data, format=_format, embed=True))

    pass


class SetupR():
    """Set up rpy2 class object for interacting with R."""
    def __init__(self, home=None, ld_library_path=''):
        # connect rpy2 with R
        self.r = _setup_r(home=home, ld_library_path=ld_library_path)
        self.home = os.environ['R_HOME']
        self.ld_library_path = os.environ['LD_LIBRARY_PATH']
        
        # set up R info
        d = {}
        for line in self.r('R.version').__str__().split('\n'):
            if len(line.split()) > 1:
                key = line[:15].split('  ')[0].replace('version.string ', 'version_string').replace(' ', '_')
                val = line[15:].split('  ')[0]
                setattr(self, key, val)
                d[key] = val
        self.version = self.version_string.split()[2]
        self.info = d
        
        # set up aliases
        self.r('len = length')
        self.r('uni = unique')
        self.r('sorted = sort')
        self.r('cd = setwd')
        pass
    
    def __call__(self, arg):
        return self.r(arg)
    
    def __cmp__(self, other):
        return self.session_info().__str__() == other.session_info().__str__()
    
    def __repr__(self):
        return str({'R_HOME' : self.home, 'LD_LIBRARY_PATH' : self.ld_library_path})
    
    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError as e:
            """Allow passing of R functions as an instance method when method is not defined here.
            
            Notes
            -----
            - will not work if R function has '.' in name - eg data.frame, as.vector
                - workaround = R('dataframe = data.frame'); R.dataframe('a=rep(0,5), b=rep(0,5)')
            
            Examples
            --------
            R.rep(0,5)
            R.colnames('df')
            R.length('1:5')
            R.length('colnames(df)')
            """
            def wrapper(*args, **kwargs):
                text = f"{attr}(%s)" % ','.join(map(str, args))

                texts = []
                for kwarg,arg in kwargs.items():
                    if isinstance(arg, str):
                        arg = f"'{arg}'"
                    texts.append(f"{kwarg}={arg}")
                if len(texts) > 0:
                    if len(args) > 0:
                        text = text.replace(")", ",%s)" % ','.join(texts))
                    else:
                        text = text.replace(")", "%s)" % ','.join(texts))
                return self.r(text)
            return wrapper
        pass
    
    def help(self, obj):
        """Get help documentation for an object, `obj`."""
        print(self.r(f'help({obj})'))
    
    def session_info(self):
        """Get R session info."""
        return print(self.r('sessionInfo()'))
    
    def library(self, lib):
        """Load library into R namespace."""
        self.r(f'library({lib})')
        pass

    def data(self, dataname, _return=False):
        """Load data object, `dataname` into R namespace."""
        self.r(f'data({dataname})')
        if _return is True:
            return self.r(dataname)
        pass

    def source(self, path):
        """Source R script."""
        self.r(f'source("{path}")')
        pass
    
    def remove(self, *args, all=False):
        """Delete objects or functions from namespace."""
        if all is True:
            self.r('remove(list=ls())')
        else:
            for arg in args:
                self.r(f'remove({arg})')
        pass
    
    def plot(self, obj, kind='plot', width=600, height=600, dpi=100, saveloc=None, **kwargs):
        """Plot in line with jupyter notebook.
        
        Parameters
        ----------
        obj - rpy2 object, or string - if string then look for unstrung string in R environment
        kind - eg plot, barplot, boxplot
        width/height/dpi - figure qualities; dpi applies to non-pdf
        saveloc - path to save figure - must contain suffix (eg .png, .jpeg, .pdf)
        
        Notes
        -----
        - this is a bit hacky (tmp files, inner()) so that I can save and show pdfs in the same process
            - without this, I can create and display non-pdfs
        """
        graphics = importr('graphics')
        
        if isinstance(obj, str):
            obj = self.r(obj)
        
        def inner(obj, width=None, height=None, dpi=None, saveloc=None, **kwargs):            
            if saveloc is None:
                saveloc = tempfile.mktemp() + '.png'
                suffix = 'png'
            else:
                print(ColorText('saved to:').bold(), saveloc)
                suffix = saveloc.split('.')[-1]  # eg pdf or png
                print(f'{suffix = }')

            with _r_inline_plot(saveloc, _format=suffix, width=width, height=height, dpi=dpi):
                self.r(f'{suffix}("{saveloc}")')  # to save (turning on will save pdf but break image in jupyter)
                graphics.__dict__[kind](obj, **kwargs)

            return saveloc

        saveloc = inner(obj, width=width, height=height, dpi=dpi, saveloc=saveloc, **kwargs) 
        if saveloc.endswith('pdf'):
            # hacky way to see an image in a notebook when saving pdf - this calls the same function so it uses a tmp.png file
            saveloc = inner(obj, kind=kind, width=width, height=height, dpi=dpi, saveloc=None, **kwargs)
            
        display(Image(filename=saveloc))
        
        if saveloc.startswith('/tmp') and op.exists(saveloc):
            os.remove(saveloc)
            
        pass

    pass

