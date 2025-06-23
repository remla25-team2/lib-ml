"""
lib-ml: text-preprocessing helper for the REMLA
sentiment-analysis.
"""

from .preprocessing import TextPreprocessor
from lib_version.version_util import VersionUtil

__version__ = VersionUtil.get_version()
__all__ = ["TextPreprocessor"]
