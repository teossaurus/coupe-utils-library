from distutils.core import setup
setup(
  name = 'coupe-utils',         # How you named your package folder (MyLib)
  packages = ['coupe-utils'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library with utilities for projects.',   # Give a short description about your library
  author = 'Teo Soares',                   # Type in your name
  author_email = 'hello@teo.works',      # Type in your E-Mail
  url = 'https://github.com/teossaurus/coupe-utils-library',   # Provide either the link to your github or to your website
  keywords = [],   # Keywords that define your package best
  install_requires=[
      'annotated-types',
      'anthropic',
      'anyio',
      'attrs',
      'Authlib',
      'beautifulsoup4',
      'bs4',
      'cachetools',
      'certifi',
      'cffi',
      'charset-normalizer',
      'cryptography',
      'distro',
      'docstring_parser',
      'exceptiongroup',
      'filelock',
      'filetype',
      'fsspec',
      'google-api-core',
      'google-auth',
      'google-cloud-aiplatform',
      'google-cloud-bigquery',
      'google-cloud-core',
      'google-cloud-firestore',
      'google-cloud-resource-manager',
      'google-cloud-storage',
      'google-cloud-tasks',
      'google-crc32c',
      'google-resumable-media',
      'googleapis-common-protos',
      'grpc-google-iam-v1',
      'grpcio',
      'grpcio-health-checking',
      'grpcio-status',
      'grpcio-tools',
      'h11',
      'httpcore',
      'httpx',
      'huggingface-hub',
      'idna',
      'jiter',
      'json5',
      'numpy',
      'openai',
      'outcome',
      'packaging',
      'proto-plus',
      'protobuf',
      'pyasn1',
      'pyasn1_modules',
      'pycparser',
      'pydantic',
      'pydantic_core',
      'PySocks',
      'python-dateutil',
      'PyYAML',
      'requests',
      'rsa',
      'selenium',
      'shapely',
      'six',
      'sniffio',
      'sortedcontainers',
      'soupsieve',
      'tokenizers',
      'tqdm',
      'trio',
      'trio-websocket',
      'typing_extensions',
      'urllib3',
      'validators',
      'vertexai',
      'weaviate-client',
      'websocket-client',
      'wsproto',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)