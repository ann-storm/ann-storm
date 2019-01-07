from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import numpy

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from custom_input_storm_twocolor import read_data_sets
