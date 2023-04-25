import os
import shutil
import pathlib
import logging
import tempfile
import subprocess
import typing as t

import click
import jinja2 as j2
import numpy as np


PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
EXAMPLES_PATH = os.path.join(PATH, 'examples')
TEMPLATES_PATH = os.path.join(PATH, 'templates')
DATASETS_PATH = os.path.join(PATH, 'datasets')

# Use this jinja2 environment to conveniently load the jinja templates which are defined as files within the
# "templates" folder of the package!
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_PATH),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'enumerate': enumerate
})

# This logger can be conveniently used as the default argument for any function which optionally accepts
# a logger. This logger will simply delete all the messages passed to it.
NULL_LOGGER = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())


# == CLI RELATED ==

def get_version():
    """
    Returns the version of the software, as dictated by the "VERSION" file of the package.
    """
    with open(VERSION_PATH) as file:
        content = file.read()
        return content.replace(' ', '').replace('\n', '')


# https://click.palletsprojects.com/en/8.1.x/api/#click.ParamType
class CsvString(click.ParamType):

    name = 'csv_string'

    def convert(self, value, param, ctx) -> t.List[str]:
        if isinstance(value, list):
            return value

        else:
            return value.split(',')


# == LATEX RELATED ==
# These functions are meant to provide a starting point for custom latex rendering. That is rendering latex
# from python strings, which were (most likely) dynamically generated based on some kind of experiment data

def render_latex(kwargs: dict,
                 output_path: str,
                 template_name: str = 'article.tex.j2'
                 ) -> None:
    """
    Renders a latex template into a PDF file. The latex template to be rendered must be a valid jinja2
    template file within the "templates" folder of the package and is identified by the string file name
    `template_name`. The argument `kwargs` is a dictionary which will be passed to that template during the
    rendering process. The designated output path of the PDF is to be given as the string absolute path
    `output_path`.

    **Example**

    The default template for this function is "article.tex.j2" which defines all the necessary boilerplate
    for an article class document. It accepts only the "content" kwargs element which is a string that is
    used as the body of the latex document.

    .. code-block:: python

        import os
        output_path = os.path.join(os.getcwd(), "out.pdf")
        kwargs = {"content": "$\text{I am a math string! } \pi = 3.141$"
        render_latex(kwargs, output_path)

    :raises ChildProcessError: if there was ANY problem with the "pdflatex" command which is used in the
        background to actually render the latex

    :param kwargs:
    :param output_path:
    :param template_name:
    :return:
    """
    with tempfile.TemporaryDirectory() as temp_path:
        # First of all we need to create the latex file on which we can then later invoke "pdflatex"
        template = TEMPLATE_ENV.get_template(template_name)
        latex_string = template.render(**kwargs)
        latex_file_path = os.path.join(temp_path, 'main.tex')
        with open(latex_file_path, mode='w') as file:
            file.write(latex_string)

        # Now we invoke the system "pdflatex" command
        command = (f'pdflatex  '
                   f'-interaction=nonstopmode '
                   f'-output-format=pdf '
                   f'-output-directory={temp_path} '
                   f'{latex_file_path} ')
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise ChildProcessError(f'pdflatex command failed! Maybe pdflatex is not properly installed on '
                                    f'the system? Error: {proc.stdout.decode()}')

        # Now finally we copy the pdf file - currently in the temp folder - to the final destination
        pdf_file_path = os.path.join(temp_path, 'main.pdf')
        shutil.copy(pdf_file_path, output_path)


# == MISC ==

def resolve_callbacks_nested_dict(data: dict) -> dict:
    """
    Given a dictionary, this function resolves all the callbacks within it in a nested manner. Every
    key that ends with the suffix "_cb" will be interpreted as some sort of callable to be executed. In the
    resolved version returned by this function the corresponding keys will be stripped of this suffix and
    the values are replaced by the actual returned value resulting from the execution of those callables.
    """
    resolved = {}
    for key, value in data.items():
        if key.endswith('_cb'):
            resolved[key[:-3]] = value()
        elif isinstance(value, dict):
            resolved[key] = resolve_callbacks_nested_dict(value)
        else:
            resolved[key] = value

    return resolved


def list_to_batches(seq: list, batch_size: int = 100) -> t.List[list]:
    batches = []
    current = []
    for i, value in enumerate(seq):
        current.append(value)

        if (i + 1) % batch_size == 0:
            batches.append(current)
            current = []

    if len(current) != 0:
        batches.append(current)

    return batches


def clean_binary_edge_importances(edge_importances: np.ndarray,
                                  edge_indices: np.ndarray,
                                  node_importances: np.ndarray,
                                  ) -> np.ndarray:
    num_channels = edge_importances.shape[1]
    for e, (i, j) in enumerate(edge_indices):
        for k in range(num_channels):
            if not node_importances[i][k] and not node_importances[j][k]:
                edge_importances[e][k] = 0

    return edge_importances
