import os
import re

from setuptools import setup

from cvextend.__init__ import __version__


def _parse_requirements(filepath):
    with open(filepath, mode='r') as f_p:
        rqr_file = f_p.readlines()

    rqr_cl = [x for x in rqr_file
              if not (re.match(r'^\s*$', x) or re.match(r'^#', x))]
    rqr = [re.sub('==', '~=', x.replace('\n', '')) for x in rqr_cl]

    return rqr


def _parse_readme(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    return content


__PROJECT_NAME__ = 'cvextend'
__DESCRIPTION__ = "Tools to extend sklearn's cross-validation classes and functions"
__AUTHORS__ = 'Lyubomir Danov'
__URL__ = 'https://github.com/ldanov/pypkg_cvextend'
__PACKAGES__ = [__PROJECT_NAME__]
_REQ_LOC = os.path.join(os.path.dirname(__file__), 'requirements.txt')
__INSTALL_REQUIRES__ = _parse_requirements(_REQ_LOC)
_RME_LOC = os.path.join(os.path.dirname(__file__), 'README.md')
__README_CONTENT__ = _parse_readme(_RME_LOC)

setup(name=__PROJECT_NAME__,
      version=__version__,
      description=__DESCRIPTION__,
      author=__AUTHORS__,
      url=__URL__,
      packages=__PACKAGES__,
      install_requires=__INSTALL_REQUIRES__,
      long_description=__README_CONTENT__,
      long_description_content_type='text/markdown'
      )
