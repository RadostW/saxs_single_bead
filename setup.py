from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='saxs_single_bead',
      version='0.0.2',
      description='All Python package to compute small angle X-ray scattering (SAXS) profiles in one-bead-per-residue approximation with numpy',
      url='https://github.com/RadostW/saxs_single_bead/',
      author='Radost Waszkiewicz',
      author_email='radost.waszkiewicz@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
      project_urls = {
          'Documentation': 'https://saxs_single_bead.readthedocs.io',
          'Source': 'https://github.com/RadostW/saxs_single_bead/'
      },
      license='MIT',
      packages=['saxs_single_bead'],
      zip_safe=False)
