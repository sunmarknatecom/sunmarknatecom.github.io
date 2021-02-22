---
layout: post
title: submission to kaggle
---

```python
!pip list
```

    Package                        Version             Location
    ------------------------------ ------------------- --------------
    absl-py                        0.10.0
    adal                           1.2.5
    affine                         2.3.0
    aiobotocore                    1.2.0
    aiohttp                        3.7.3
    aiohttp-cors                   0.7.0
    aioitertools                   0.7.1
    aioredis                       1.3.1
    albumentations                 0.5.2
    alembic                        1.5.2
    allennlp                       1.3.0
    altair                         4.1.0
    annoy                          1.17.0
    ansiwrap                       0.8.4
    appdirs                        1.4.4
    argon2-cffi                    20.1.0
    arrow                          0.17.0
    arviz                          0.11.0
    asn1crypto                     1.4.0
    astropy                        4.2
    astunparse                     1.6.3
    async-generator                1.10
    async-timeout                  3.0.1
    attrs                          20.3.0
    audioread                      2.1.9
    autocfg                        0.0.6
    autogluon.core                 0.0.16b20210126
    autograd                       1.3
    Babel                          2.9.0
    backcall                       0.2.0
    backports.functools-lru-cache  1.6.1
    basemap                        1.2.1
    bayesian-optimization          1.2.0
    bayespy                        0.5.20
    bcrypt                         3.2.0
    binaryornot                    0.4.4
    biopython                      1.78
    black                          20.8b1
    bleach                         3.2.1
    blessings                      1.7
    blinker                        1.4
    blis                           0.7.4
    bokeh                          2.2.3
    Boruta                         0.3
    boto3                          1.16.60
    botocore                       1.19.60
    Bottleneck                     1.3.2
    bq-helper                      0.4.1               /src/bq-helper
    bqplot                         0.12.21
    branca                         0.4.2
    brewer2mpl                     1.4.1
    brotlipy                       0.7.0
    cachetools                     4.1.1
    caip-notebooks-serverextension 1.0.0
    Cartopy                        0.18.0
    catalogue                      1.0.0
    catalyst                       20.12
    catboost                       0.24.4
    category-encoders              2.2.2
    certifi                        2020.12.5
    cesium                         0.9.12
    cffi                           1.14.4
    cftime                         1.3.1
    chardet                        3.0.4
    cleverhans                     3.0.1
    click                          7.1.2
    click-plugins                  1.1.1
    cliff                          3.6.0
    cligj                          0.7.1
    cloud-tpu-client               0.10
    cloudpickle                    1.6.0
    cmaes                          0.7.0
    cmd2                           1.4.0
    cmdstanpy                      0.9.5
    cmudict                        0.4.5
    colorama                       0.4.4
    colorcet                       2.0.6
    colorful                       0.5.4
    colorlog                       4.7.2
    colorlover                     0.3.0
    conda                          4.9.2
    conda-package-handling         1.7.2
    configparser                   5.0.1
    ConfigSpace                    0.4.16
    confuse                        1.4.0
    contextily                     1.0.1
    contextlib2                    0.6.0.post1
    convertdate                    2.2.0
    cookiecutter                   1.7.2
    cryptography                   3.3.1
    cudf                           0.16.0
    cufflinks                      0.17.3
    cuml                           0.16.0
    cupy                           8.3.0
    cupy-cuda102                   8.3.0
    CVXcanon                       0.1.2
    cvxpy                          1.1.7
    cycler                         0.10.0
    cymem                          2.0.5
    cysignals                      1.10.2
    Cython                         0.29.21
    cytoolz                        0.11.0
    dask                           2021.1.1
    dask-cudf                      0.16.0
    dataclasses                    0.6
    datashader                     0.12.0
    datashape                      0.5.2
    datatable                      0.11.1
    deap                           1.3.1
    decorator                      4.4.2
    decord                         0.4.2
    deepdish                       0.3.6
    defusedxml                     0.6.0
    Delorean                       1.0.0
    Deprecated                     1.2.10
    descartes                      1.1.0
    dill                           0.3.3
    dipy                           1.3.0
    distributed                    2021.1.1
    dlib                           19.21.1
    dm-tree                        0.1.5
    docker                         4.4.1
    docker-pycreds                 0.4.0
    docutils                       0.16
    earthengine-api                0.1.248
    easydev                        0.10.1
    ecos                           2.0.7.post1
    eli5                           0.11.0
    emoji                          1.1.0
    en-core-web-lg                 2.3.1
    en-core-web-sm                 2.3.1
    entrypoints                    0.3
    ephem                          3.7.7.1
    essentia                       2.1b6.dev374
    fancyimpute                    0.5.5
    fastai                         2.2.5
    fastavro                       1.3.0
    fastcore                       1.3.19
    fastprogress                   1.0.0
    fastrlock                      0.5
    fasttext                       0.9.2
    fbpca                          1.0
    fbprophet                      0.7.1
    feather-format                 0.4.1
    featuretools                   0.23.0
    filelock                       3.0.12
    Fiona                          1.8.18
    fitter                         1.3.0
    flake8                         3.8.4
    flashtext                      2.7
    Flask                          1.1.2
    flatbuffers                    1.12
    folium                         0.12.1
    fsspec                         0.8.5
    funcy                          1.15
    fury                           0.6.1
    future                         0.18.2
    fuzzywuzzy                     0.18.0
    gast                           0.3.3
    gatspy                         0.3
    gcsfs                          0.7.1
    GDAL                           3.1.4
    gensim                         3.8.3
    geographiclib                  1.50
    Geohash                        1.0
    geojson                        2.5.0
    geopandas                      0.8.2
    geoplot                        0.4.1
    geopy                          2.1.0
    geoviews                       1.8.1
    ggplot                         0.11.5
    gitdb                          4.0.5
    GitPython                      3.1.12
    gluoncv                        0.9.1.post0
    gluonnlp                       0.10.0
    google-api-core                1.22.4
    google-api-python-client       1.8.0
    google-auth                    1.24.0
    google-auth-httplib2           0.0.4
    google-auth-oauthlib           0.4.2
    google-cloud-automl            1.0.1
    google-cloud-bigquery          1.12.1
    google-cloud-bigquery-storage  1.0.0
    google-cloud-bigtable          1.4.0
    google-cloud-core              1.3.0
    google-cloud-dataproc          1.1.1
    google-cloud-datastore         1.12.0
    google-cloud-firestore         1.8.1
    google-cloud-kms               1.4.0
    google-cloud-language          2.0.0
    google-cloud-logging           1.15.1
    google-cloud-monitoring        1.1.0
    google-cloud-pubsub            1.7.0
    google-cloud-scheduler         1.3.0
    google-cloud-spanner           1.17.1
    google-cloud-speech            1.3.2
    google-cloud-storage           1.30.0
    google-cloud-tasks             1.5.0
    google-cloud-translate         3.0.2
    google-cloud-videointelligence 2.0.0
    google-cloud-vision            2.1.0
    google-crc32c                  1.1.0
    google-pasta                   0.2.0
    google-resumable-media         1.2.0
    googleapis-common-protos       1.52.0
    gplearn                        0.4.1
    gpustat                        0.6.0
    gpxpy                          1.4.2
    graphviz                       0.8.4
    grpc-google-iam-v1             0.12.3
    grpcio                         1.32.0
    grpcio-gcp                     0.2.2
    gym                            0.18.0
    h2o                            3.32.0.3
    h5py                           2.10.0
    haversine                      2.3.0
    HeapDict                       1.0.1
    hep-ml                         0.6.2
    hiredis                        1.1.0
    hmmlearn                       0.2.4
    holidays                       0.10.4
    holoviews                      1.14.1
    hpsklearn                      0.1.0
    html5lib                       1.1
    htmlmin                        0.1.12
    httplib2                       0.18.1
    httplib2shim                   0.0.3
    humanize                       3.2.0
    hunspell                       0.5.5
    husl                           4.0.3
    hyperopt                       0.2.5
    hypertools                     0.6.3
    hypothesis                     6.0.3
    ibis-framework                 1.4.0
    idna                           2.10
    imagecodecs                    2021.1.11
    ImageHash                      4.2.0
    imageio                        2.9.0
    imbalanced-learn               0.7.0
    imgaug                         0.4.0
    implicit                       0.4.4
    importlib-metadata             3.3.0
    iniconfig                      1.1.1
    ipykernel                      5.1.1
    ipython                        7.19.0
    ipython-genutils               0.2.0
    ipython-sql                    0.3.9
    ipywidgets                     7.6.2
    iso3166                        1.0.1
    isoweek                        1.3.3
    itsdangerous                   1.1.0
    Janome                         0.4.1
    jax                            0.2.6
    jaxlib                         0.1.57+cuda102
    jedi                           0.18.0
    jieba                          0.42.1
    Jinja2                         2.11.2
    jinja2-time                    0.2.0
    jmespath                       0.10.0
    joblib                         1.0.0
    json5                          0.9.5
    jsonnet                        0.17.0
    jsonpickle                     1.5.0
    jsonschema                     3.2.0
    jupyter                        1.0.0
    jupyter-aihub-deploy-extension 0.2
    jupyter-client                 6.1.7
    jupyter-console                6.2.0
    jupyter-core                   4.7.0
    jupyter-http-over-ws           0.0.8
    jupyterlab                     1.2.16
    jupyterlab-git                 0.11.0
    jupyterlab-pygments            0.1.2
    jupyterlab-server              1.2.0
    jupyterlab-widgets             1.0.0
    kaggle                         1.5.10
    kaggle-environments            1.7.7
    Keras                          2.4.3
    Keras-Preprocessing            1.1.2
    keras-tuner                    1.0.2
    kiwisolver                     1.3.1
    kmapper                        1.4.1
    kmodes                         0.10.2
    knnimpute                      0.1.0
    korean-lunar-calendar          0.2.1
    kornia                         0.4.1
    kubernetes                     12.0.1
    langid                         1.1.6
    learntools                     0.3.4
    leven                          1.0.4
    libcst                         0.3.16
    librosa                        0.8.0
    lightfm                        1.16
    lightgbm                       3.1.1
    lime                           0.2.0.1
    line-profiler                  3.1.0
    llvmlite                       0.35.0
    lml                            0.1.0
    locket                         0.2.1
    LunarCalendar                  0.0.9
    lxml                           4.6.2
    Mako                           1.1.4
    mapclassify                    2.4.2
    marisa-trie                    0.7.5
    Markdown                       3.3.3
    markovify                      0.8.3
    MarkupSafe                     1.1.1
    matplotlib                     3.3.3
    matplotlib-venn                0.11.6
    mccabe                         0.6.1
    memory-profiler                0.58.0
    mercantile                     1.1.6
    missingno                      0.4.2
    mistune                        0.8.4
    mizani                         0.7.2
    ml-metrics                     0.1.4
    mlcrate                        0.2.0
    mlens                          0.2.3
    mlxtend                        0.18.0
    mmh3                           2.5.1
    mne                            0.22.0
    mnist                          0.2.2
    mock                           4.0.3
    mpld3                          0.5.2
    mpmath                         1.1.0
    msgpack                        1.0.2
    msgpack-numpy                  0.4.7.1
    multidict                      5.1.0
    multimethod                    1.4
    multipledispatch               0.6.0
    multiprocess                   0.70.11.1
    munch                          2.5.0
    murmurhash                     1.0.5
    mxnet-cu102                    1.7.0
    mypy-extensions                0.4.3
    nb-conda                       2.2.1
    nb-conda-kernels               2.3.1
    nbclient                       0.5.1
    nbconvert                      6.0.7
    nbdime                         2.1.0
    nbformat                       5.0.8
    nest-asyncio                   1.4.3
    netCDF4                        1.5.5.1
    networkx                       2.5
    nibabel                        3.2.1
    nilearn                        0.7.0
    nltk                           3.2.4
    nnabla                         1.13.0
    nnabla-ext-cuda102             1.13.0
    nose                           1.3.7
    notebook                       5.5.0
    notebook-executor              0.2
    numba                          0.52.0
    numexpr                        2.7.2
    numpy                          1.19.5
    nvidia-ml-py3                  7.352.0
    oauth2client                   4.1.3
    oauthlib                       3.0.1
    odfpy                          1.4.1
    olefile                        0.46
    onnx                           1.8.0
    opencensus                     0.7.12
    opencensus-context             0.1.2
    opencv-python                  4.5.1.48
    opencv-python-headless         4.5.1.48
    openslide-python               1.1.2
    opt-einsum                     3.3.0
    optuna                         2.4.0
    orderedmultidict               1.0.1
    ortools                        8.1.8487
    osmnx                          1.0.1
    osqp                           0.6.2.post0
    overrides                      3.1.0
    packaging                      20.8
    palettable                     3.3.0
    pandas                         1.1.5
    pandas-datareader              0.9.0
    pandas-profiling               2.8.0
    pandas-summary                 0.0.7
    pandasql                       0.7.3
    pandocfilters                  1.4.2
    panel                          0.10.3
    papermill                      2.2.2
    param                          1.10.1
    paramiko                       2.7.2
    parso                          0.8.1
    partd                          1.1.0
    path                           15.0.1
    path.py                        12.5.0
    pathos                         0.2.7
    pathspec                       0.8.1
    pathtools                      0.1.2
    patsy                          0.5.1
    pbr                            5.5.1
    pdf2image                      1.14.0
    PDPbox                         0.2.0+13.g73c6966
    pexpect                        4.8.0
    phik                           0.10.0
    pickleshare                    0.7.5
    Pillow                         7.2.0
    pip                            21.0
    plac                           1.1.3
    plotly                         4.14.3
    plotly-express                 0.4.1
    plotnine                       0.7.1
    pluggy                         0.13.1
    polyglot                       16.7.4
    pooch                          1.3.0
    portalocker                    2.1.0
    pox                            0.2.9
    poyo                           0.5.0
    ppca                           0.0.4
    ppft                           1.6.6.3
    preprocessing                  0.1.13
    preshed                        3.0.5
    prettytable                    0.7.2
    prometheus-client              0.9.0
    promise                        2.3
    prompt-toolkit                 3.0.8
    pronouncing                    0.2.0
    proto-plus                     1.13.0
    protobuf                       3.14.0
    psutil                         5.8.0
    ptyprocess                     0.7.0
    pudb                           2020.1
    py                             1.10.0
    py-lz4framed                   0.14.0
    py-spy                         0.3.4
    py-stringmatching              0.4.2
    py-stringsimjoin               0.3.2
    pyaml                          20.4.0
    PyArabic                       0.6.10
    pyarrow                        1.0.1
    pyasn1                         0.4.8
    pyasn1-modules                 0.2.7
    PyAstronomy                    0.15.2
    pybind11                       2.6.1
    pycairo                        1.20.0
    pycodestyle                    2.6.0
    pycosat                        0.6.3
    pycountry                      20.7.3
    pycparser                      2.20
    pycrypto                       2.6.1
    pyct                           0.4.8
    pycuda                         2020.1
    pydash                         4.9.2
    pydegensac                     0.1.2
    pydicom                        2.1.2
    pydub                          0.24.1
    pyemd                          0.5.1
    pyerfa                         1.7.1.1
    pyexcel-io                     0.6.4
    pyexcel-ods                    0.6.0
    pyfasttext                     0.4.6
    pyflakes                       2.2.0
    pyglet                         1.5.0
    Pygments                       2.7.3
    PyJWT                          1.7.1
    pykalman                       0.9.5
    pyLDAvis                       2.1.2
    pymc3                          3.11.0
    PyMeeus                        0.3.7
    pymongo                        3.11.2
    Pympler                        0.9
    PyNaCl                         1.4.0
    pynndescent                    0.5.1
    pynvrtc                        9.2
    pyocr                          0.8
    pyOpenSSL                      20.0.1
    pyparsing                      2.4.7
    pyPdf                          1.13
    pyperclip                      1.8.1
    PyPrind                        2.11.2
    pyproj                         2.6.1.post1
    PyQt5                          5.12.3
    PyQt5-sip                      4.19.18
    PyQtChart                      5.12
    PyQtWebEngine                  5.12.1
    pyrsistent                     0.17.3
    pysal                          2.1.0
    pyshp                          2.1.3
    PySocks                        1.7.1
    pystan                         2.19.1.1
    pytesseract                    0.3.7
    pytest                         6.2.2
    pytext-nlp                     0.1.2
    python-dateutil                2.8.1
    python-editor                  1.0.4
    python-igraph                  0.8.3
    python-Levenshtein             0.12.1
    python-louvain                 0.15
    python-slugify                 4.0.1
    pytools                        2021.1
    pytorch-ignite                 0.4.2
    pytorch-lightning              1.1.5
    pytz                           2019.3
    PyUpSet                        0.1.1.post7
    pyviz-comms                    2.0.1
    PyWavelets                     1.1.1
    PyYAML                         5.3.1
    pyzmq                          20.0.0
    qdldl                          0.1.5.post0
    qgrid                          1.3.1
    qtconsole                      5.0.2
    QtPy                           1.9.0
    randomgen                      1.16.6
    rasterio                       1.2.0
    ray                            1.1.0
    redis                          3.5.3
    regex                          2020.11.13
    requests                       2.25.1
    requests-oauthlib              1.3.0
    resampy                        0.2.2
    retrying                       1.3.3
    rgf-python                     3.9.0
    rmm                            0.16.0
    rsa                            4.6
    Rtree                          0.9.7
    ruamel-yaml-conda              0.15.80
    s2sphere                       0.2.5
    s3fs                           0.5.2
    s3transfer                     0.3.4
    sacremoses                     0.0.43
    scattertext                    0.1.0.0
    scikit-image                   0.18.1
    scikit-learn                   0.23.2
    scikit-multilearn              0.2.0
    scikit-optimize                0.8.1
    scikit-plot                    0.3.7
    scikit-surprise                1.1.1
    scipy                          1.4.1
    scs                            2.1.2
    seaborn                        0.11.1
    Send2Trash                     1.5.0
    sentencepiece                  0.1.95
    sentry-sdk                     0.19.5
    setuptools                     49.6.0.post20201009
    setuptools-git                 1.2
    shap                           0.38.1
    Shapely                        1.7.1
    shortuuid                      1.0.1
    SimpleITK                      2.0.2
    simplejson                     3.17.2
    six                            1.15.0
    sklearn-contrib-py-earth       0.1.0+1.gdde5f89
    sklearn-pandas                 2.0.4
    slicer                         0.0.7
    smart-open                     4.1.2
    smhasher                       0.150.1
    smmap                          3.0.4
    snuggs                         1.4.7
    sortedcontainers               2.3.0
    SoundFile                      0.10.3.post1
    spacy                          2.3.5
    spectral                       0.22.1
    sphinx-rtd-theme               0.2.4
    SQLAlchemy                     1.3.22
    sqlparse                       0.4.1
    squarify                       0.4.3
    srsly                          1.0.5
    statsmodels                    0.12.1
    stemming                       1.0.1
    stevedore                      3.3.0
    stop-words                     2018.7.23
    stopit                         1.1.2
    subprocess32                   3.5.4
    sympy                          1.7.1
    tables                         3.6.1
    tabulate                       0.8.7
    tangled-up-in-unicode          0.0.6
    tblib                          1.7.0
    tenacity                       6.3.1
    tensorboard                    2.4.1
    tensorboard-plugin-wit         1.8.0
    tensorboardX                   2.1
    tensorflow                     2.4.0
    tensorflow-addons              0.12.0
    tensorflow-cloud               0.1.11
    tensorflow-datasets            3.0.0
    tensorflow-estimator           2.4.0
    tensorflow-gcs-config          2.1.7
    tensorflow-hub                 0.11.0
    tensorflow-metadata            0.27.0
    tensorflow-probability         0.12.1
    Tensorforce                    0.5.5
    tensorpack                     0.11
    termcolor                      1.1.0
    terminado                      0.9.2
    terminaltables                 3.1.0
    testpath                       0.4.4
    text-unidecode                 1.3
    textblob                       0.15.3
    texttable                      1.6.3
    textwrap3                      0.9.2
    Theano                         1.0.5
    Theano-PyMC                    1.1.0
    thinc                          7.4.5
    threadpoolctl                  2.1.0
    tifffile                       2021.1.14
    tokenizers                     0.9.4
    toml                           0.10.2
    toolz                          0.11.1
    torch                          1.7.0
    torchaudio                     0.7.0a0+ac17b64
    torchtext                      0.8.0a0+cd6902d
    torchvision                    0.8.1
    tornado                        5.0.2
    TPOT                           0.11.7
    tqdm                           4.55.1
    traitlets                      5.0.5
    traittypes                     0.2.1
    transformers                   4.0.1
    treelite                       0.93
    treelite-runtime               0.93
    trueskill                      0.4.5
    tsfresh                        0.17.0
    typed-ast                      1.4.2
    typeguard                      2.10.0
    typing-extensions              3.7.4.3
    typing-inspect                 0.6.0
    tzlocal                        2.1
    ucx-py                         0.16.0
    umap-learn                     0.5.0
    Unidecode                      1.1.2
    update-checker                 0.18.0
    uritemplate                    3.0.1
    urllib3                        1.26.2
    urwid                          2.1.2
    vecstack                       0.4.0
    visions                        0.4.4
    vowpalwabbit                   8.9.0
    vtk                            9.0.1
    Wand                           0.5.3
    wandb                          0.10.15
    wasabi                         0.8.1
    watchdog                       0.10.4
    wavio                          0.0.4
    wcwidth                        0.2.5
    webencodings                   0.5.1
    websocket-client               0.57.0
    Werkzeug                       1.0.1
    wfdb                           3.2.0
    wheel                          0.36.2
    whichcraft                     0.6.1
    widgetsnbextension             3.5.1
    Wordbatch                      1.4.6
    wordcloud                      1.8.1
    wordsegment                    1.3.1
    wrapt                          1.12.1
    xarray                         0.16.2
    xgboost                        1.3.3
    xvfbwrapper                    0.2.9
    yacs                           0.1.8
    yarl                           1.6.3
    yellowbrick                    1.2.1
    zict                           2.0.0
    zipp                           3.4.0



```python
!python --version
```

    Python 3.7.9



```python
import time

start = time.time()
```

# DATA


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
```


```python
# Root pathway declaration

base_path = '/kaggle/input/ranzcr-clip-catheter-line-classification/'
```


```python
# print the size of train, test

for path in ['train','test']:
    print('{} data size:{}'.format(path,len(os.listdir(os.path.join(base_path,path)))))
```

    train data size:30083
    test data size:3582



```python
df_train = pd.read_csv(os.path.join(base_path,'train.csv'))
df_test = pd.read_csv(os.path.join(base_path,'sample_submission.csv'))

Labels = np.array(df_train.drop(['StudyInstanceUID','PatientID'],axis=1).columns)

print('train_csv shapes',df_train.shape)
print('sample_submission_csv shapes',df_test.shape)
print()
print('Labels:',Labels)
```

    train_csv shapes (30083, 13)
    sample_submission_csv shapes (3582, 12)
    
    Labels: ['ETT - Abnormal' 'ETT - Borderline' 'ETT - Normal' 'NGT - Abnormal'
     'NGT - Borderline' 'NGT - Incompletely Imaged' 'NGT - Normal'
     'CVC - Abnormal' 'CVC - Borderline' 'CVC - Normal'
     'Swan Ganz Catheter Present']



```python
display(df_train.tail(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StudyInstanceUID</th>
      <th>ETT - Abnormal</th>
      <th>ETT - Borderline</th>
      <th>ETT - Normal</th>
      <th>NGT - Abnormal</th>
      <th>NGT - Borderline</th>
      <th>NGT - Incompletely Imaged</th>
      <th>NGT - Normal</th>
      <th>CVC - Abnormal</th>
      <th>CVC - Borderline</th>
      <th>CVC - Normal</th>
      <th>Swan Ganz Catheter Present</th>
      <th>PatientID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30073</th>
      <td>1.2.826.0.1.3680043.8.498.44675490137018694724...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>b32884471</td>
    </tr>
    <tr>
      <th>30074</th>
      <td>1.2.826.0.1.3680043.8.498.60620865844062547094...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>45d82b916</td>
    </tr>
    <tr>
      <th>30075</th>
      <td>1.2.826.0.1.3680043.8.498.12112840402677606176...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>cccbc15ba</td>
    </tr>
    <tr>
      <th>30076</th>
      <td>1.2.826.0.1.3680043.8.498.59704742952729813362...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>172c3c7ed</td>
    </tr>
    <tr>
      <th>30077</th>
      <td>1.2.826.0.1.3680043.8.498.97304417279653947772...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>b304abf90</td>
    </tr>
    <tr>
      <th>30078</th>
      <td>1.2.826.0.1.3680043.8.498.74257566841157531124...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5b5b9ac30</td>
    </tr>
    <tr>
      <th>30079</th>
      <td>1.2.826.0.1.3680043.8.498.46510939987173529969...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7192404d8</td>
    </tr>
    <tr>
      <th>30080</th>
      <td>1.2.826.0.1.3680043.8.498.43173270582850645437...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>d4d1b066d</td>
    </tr>
    <tr>
      <th>30081</th>
      <td>1.2.826.0.1.3680043.8.498.95092491950130838685...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>01a6602b8</td>
    </tr>
    <tr>
      <th>30082</th>
      <td>1.2.826.0.1.3680043.8.498.99518162226171269731...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>e692d316c</td>
    </tr>
  </tbody>
</table>
</div>



```python
# show the figures

train_path =  str(base_path + '/train')

rows = 3
cols = 3

display_size = rows*cols
files = zip(df_train['StudyInstanceUID'][:display_size]+'.jpg',df_train.drop(['StudyInstanceUID','PatientID'],axis=1).to_numpy()[:display_size])

plt.figure(figsize=(15,10))

for i, (file_name,label) in enumerate(files):
    plt.subplot(rows,cols,i+1)
    img = cv2.imread(os.path.join(train_path,file_name),0)
    plt.imshow(img, cmap='gray')
    plt.title(Labels[[lbl[0] for lbl in np.argwhere(label==1)]].tolist())
    plt.ylabel('img_size='+str(img.shape))
plt.show()
```


![png](output_9_0.png)



```python
files = zip(df_train['StudyInstanceUID'][:display_size]+'.jpg',df_train.drop(['StudyInstanceUID','PatientID'],axis=1).to_numpy()[:display_size])
```


```python
filename = list(files)[0][0]
```


```python
image_sample = cv2.imread(os.path.join(train_path, filename),0)
```


```python
plt.imshow(image_sample)
plt.show()
```


![png](output_13_0.png)



```python
# helper function

def path2img(path, label = None, resize = (224,224), mode_rgb = 0):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 3)
    if mode_rgb == 0:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img,resize)
    img = tf.cast(img,tf.float32)/255.0
    return img, label if label!=None else img
```


```python
# helper function - input data


def process_dataset(files, training_data):
    A = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    files_ds = files_ds.map(path2img, num_parallel_calls=A)
    if training_data:
        files_ds = files_ds.cache()
    return files_ds
```


```python
# file pathway
paths = base_path + 'train/' + df_train['StudyInstanceUID']+'.jpg'
```


```python
type(paths)
```




    pandas.core.series.Series




```python
paths
```




    0        /kaggle/input/ranzcr-clip-catheter-line-classi...
    1        /kaggle/input/ranzcr-clip-catheter-line-classi...
    2        /kaggle/input/ranzcr-clip-catheter-line-classi...
    3        /kaggle/input/ranzcr-clip-catheter-line-classi...
    4        /kaggle/input/ranzcr-clip-catheter-line-classi...
                                   ...                        
    30078    /kaggle/input/ranzcr-clip-catheter-line-classi...
    30079    /kaggle/input/ranzcr-clip-catheter-line-classi...
    30080    /kaggle/input/ranzcr-clip-catheter-line-classi...
    30081    /kaggle/input/ranzcr-clip-catheter-line-classi...
    30082    /kaggle/input/ranzcr-clip-catheter-line-classi...
    Name: StudyInstanceUID, Length: 30083, dtype: object




```python
# df_train(filename) to label extraction
labels = df_train.drop(['StudyInstanceUID','PatientID'],axis=1).to_numpy()
```


```python
type(labels)
```




    numpy.ndarray




```python
labels
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 1, ..., 0, 1, 0],
           [0, 0, 0, ..., 1, 0, 0],
           ...,
           [0, 0, 1, ..., 0, 1, 0],
           [0, 0, 0, ..., 1, 0, 0],
           [0, 0, 1, ..., 0, 1, 0]])




```python
# sklearn.model_selection -> train_test_split
x_train_path, x_val_path, y_train, y_val = train_test_split(paths, labels, test_size = .2, random_state=42, shuffle = True)

print('train_size:',len(y_train))
print('val_size:',len(y_val))

train = process_dataset((x_train_path,y_train), training_data = True)
val = process_dataset((x_val_path,y_val), training_data = True)

print('total = ', len(y_train)+len(y_val))
```

    train_size: 24066
    val_size: 6017
    total =  30083



```python
type(train)
```




    tensorflow.python.data.ops.dataset_ops.CacheDataset




```python
plt.figure(figsize=(12,6))
print('train samples:')
for i, (img,label) in enumerate(train.take(2)):
    plt.subplot(1,2,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(Labels[[lbl[0] for lbl in np.argwhere(label==1)]].tolist())
    plt.xlabel('image_size:'+str(img.shape))
plt.show()

plt.figure(figsize=(12,6))
print('validation samples:')
for i, (img,label) in enumerate(val.take(2)):
    plt.subplot(1,2,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(Labels[[lbl[0] for lbl in np.argwhere(label==1)]].tolist())
    plt.xlabel('image_size:'+str(img.shape))
plt.show()
```

    train samples:



![png](output_24_1.png)


    validation samples:



![png](output_24_3.png)



```python
input_shape = img.shape
print('input_shape:',input_shape)
num_classes = len(Labels)

model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3), padding = 'same', input_shape = input_shape))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(128,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(256,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))         
model.add(MaxPooling2D(pool_size=(2,2)))   

model.add(Conv2D(512,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))         
model.add(Conv2D(512,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
model.summary()
```

    input_shape: (224, 224, 1)
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 224, 224, 64)      640       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 224, 224, 64)      256       
    _________________________________________________________________
    activation (Activation)      (None, 224, 224, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 224, 224, 64)      36928     
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 224, 224, 64)      256       
    _________________________________________________________________
    activation_1 (Activation)    (None, 224, 224, 64)      0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 112, 112, 128)     73856     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 112, 112, 128)     512       
    _________________________________________________________________
    activation_2 (Activation)    (None, 112, 112, 128)     0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 112, 112, 128)     147584    
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 112, 112, 128)     448       
    _________________________________________________________________
    activation_3 (Activation)    (None, 112, 112, 128)     0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 56, 56, 128)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 56, 56, 256)       295168    
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 56, 56, 256)       1024      
    _________________________________________________________________
    activation_4 (Activation)    (None, 56, 56, 256)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 56, 56, 256)       590080    
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 56, 56, 256)       224       
    _________________________________________________________________
    activation_5 (Activation)    (None, 56, 56, 256)       0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 28, 28, 256)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 28, 28, 512)       2048      
    _________________________________________________________________
    activation_6 (Activation)    (None, 28, 28, 512)       0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 28, 28, 512)       2048      
    _________________________________________________________________
    activation_7 (Activation)    (None, 28, 28, 512)       0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               6422784   
    _________________________________________________________________
    activation_8 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                16448     
    _________________________________________________________________
    activation_9 (Activation)    (None, 64)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    activation_10 (Activation)   (None, 32)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 11)                363       
    _________________________________________________________________
    activation_11 (Activation)   (None, 11)                0         
    =================================================================
    Total params: 11,132,715
    Trainable params: 11,129,307
    Non-trainable params: 3,408
    _________________________________________________________________



```python
lr = 7e-4
epochs = 20
batch_size = 86

train_b = train.batch(batch_size)
val_b = val.batch(batch_size)
 
CP = ModelCheckpoint('/kaggle/working/model.hdf5', save_best_only=True, verbose = 1)

model.compile(optimizer = Adam(lr=lr), loss='binary_crossentropy', metrics=['AUC'])
H = model.fit(train_b, epochs = epochs, validation_data = val_b, callbacks=[CP], shuffle=True)
```

    Epoch 1/20
    280/280 [==============================] - 2697s 10s/step - loss: 0.6627 - auc: 0.7277 - val_loss: 0.3174 - val_auc: 0.8488
    
    Epoch 00001: val_loss improved from inf to 0.31741, saving model to /kaggle/working/model.hdf5
    Epoch 2/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.3287 - auc: 0.8371 - val_loss: 0.3073 - val_auc: 0.8838
    
    Epoch 00002: val_loss improved from 0.31741 to 0.30734, saving model to /kaggle/working/model.hdf5
    Epoch 3/20
    280/280 [==============================] - 166s 591ms/step - loss: 0.3049 - auc: 0.8620 - val_loss: 0.2870 - val_auc: 0.8961
    
    Epoch 00003: val_loss improved from 0.30734 to 0.28696, saving model to /kaggle/working/model.hdf5
    Epoch 4/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2896 - auc: 0.8780 - val_loss: 0.2745 - val_auc: 0.8991
    
    Epoch 00004: val_loss improved from 0.28696 to 0.27450, saving model to /kaggle/working/model.hdf5
    Epoch 5/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2814 - auc: 0.8858 - val_loss: 0.3217 - val_auc: 0.8531
    
    Epoch 00005: val_loss did not improve from 0.27450
    Epoch 6/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2747 - auc: 0.8922 - val_loss: 0.2706 - val_auc: 0.8960
    
    Epoch 00006: val_loss improved from 0.27450 to 0.27057, saving model to /kaggle/working/model.hdf5
    Epoch 7/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2636 - auc: 0.9026 - val_loss: 0.2726 - val_auc: 0.8908
    
    Epoch 00007: val_loss did not improve from 0.27057
    Epoch 8/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2519 - auc: 0.9125 - val_loss: 0.2430 - val_auc: 0.9208
    
    Epoch 00008: val_loss improved from 0.27057 to 0.24296, saving model to /kaggle/working/model.hdf5
    Epoch 9/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2460 - auc: 0.9171 - val_loss: 0.2572 - val_auc: 0.9065
    
    Epoch 00009: val_loss did not improve from 0.24296
    Epoch 10/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2433 - auc: 0.9191 - val_loss: 0.2516 - val_auc: 0.9113
    
    Epoch 00010: val_loss did not improve from 0.24296
    Epoch 11/20
    280/280 [==============================] - 166s 591ms/step - loss: 0.2417 - auc: 0.9204 - val_loss: 0.2489 - val_auc: 0.9175
    
    Epoch 00011: val_loss did not improve from 0.24296
    Epoch 12/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2412 - auc: 0.9208 - val_loss: 0.2394 - val_auc: 0.9217
    
    Epoch 00012: val_loss improved from 0.24296 to 0.23939, saving model to /kaggle/working/model.hdf5
    Epoch 13/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2393 - auc: 0.9223 - val_loss: 0.2562 - val_auc: 0.9077
    
    Epoch 00013: val_loss did not improve from 0.23939
    Epoch 14/20
    280/280 [==============================] - 165s 590ms/step - loss: 0.2381 - auc: 0.9230 - val_loss: 0.3281 - val_auc: 0.8604
    
    Epoch 00014: val_loss did not improve from 0.23939
    Epoch 15/20
    280/280 [==============================] - 165s 590ms/step - loss: 0.2388 - auc: 0.9229 - val_loss: 0.2603 - val_auc: 0.9071
    
    Epoch 00015: val_loss did not improve from 0.23939
    Epoch 16/20
    280/280 [==============================] - 165s 590ms/step - loss: 0.2382 - auc: 0.9231 - val_loss: 0.2440 - val_auc: 0.9173
    
    Epoch 00016: val_loss did not improve from 0.23939
    Epoch 17/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2356 - auc: 0.9250 - val_loss: 0.2787 - val_auc: 0.8853
    
    Epoch 00017: val_loss did not improve from 0.23939
    Epoch 18/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2356 - auc: 0.9249 - val_loss: 0.2364 - val_auc: 0.9227
    
    Epoch 00018: val_loss improved from 0.23939 to 0.23641, saving model to /kaggle/working/model.hdf5
    Epoch 19/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2348 - auc: 0.9254 - val_loss: 0.2410 - val_auc: 0.9208
    
    Epoch 00019: val_loss did not improve from 0.23641
    Epoch 20/20
    280/280 [==============================] - 165s 591ms/step - loss: 0.2332 - auc: 0.9265 - val_loss: 0.2371 - val_auc: 0.9228
    
    Epoch 00020: val_loss did not improve from 0.23641



```python
plt.figure(figsize=(8,4))
plt.plot(H.history['auc'])
plt.plot(H.history['val_auc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

plt.figure(figsize=(8,4))
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
```


![png](output_27_0.png)



![png](output_27_1.png)



```python
del train_b
del val_b

test_path = base_path + 'test/' + df_test['StudyInstanceUID']+'.jpg'

test = process_dataset(test_path, training_data = False)

#load the best model saved
model = load_model('/kaggle/working/model.hdf5')

predictions = model.predict(test.batch(1), verbose = 1)

df_test.iloc[:,1:] = predictions

display(df_test.head())

df_test.to_csv('submission.csv', index = False)
```

    3582/3582 [==============================] - 361s 100ms/step



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StudyInstanceUID</th>
      <th>ETT - Abnormal</th>
      <th>ETT - Borderline</th>
      <th>ETT - Normal</th>
      <th>NGT - Abnormal</th>
      <th>NGT - Borderline</th>
      <th>NGT - Incompletely Imaged</th>
      <th>NGT - Normal</th>
      <th>CVC - Abnormal</th>
      <th>CVC - Borderline</th>
      <th>CVC - Normal</th>
      <th>Swan Ganz Catheter Present</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.2.826.0.1.3680043.8.498.46923145579096002617...</td>
      <td>8.404819e-03</td>
      <td>1.223760e-01</td>
      <td>0.808184</td>
      <td>0.021490</td>
      <td>0.045480</td>
      <td>0.284181</td>
      <td>0.462424</td>
      <td>0.104707</td>
      <td>0.334759</td>
      <td>0.726770</td>
      <td>0.075364</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.2.826.0.1.3680043.8.498.84006870182611080091...</td>
      <td>1.033413e-12</td>
      <td>3.378371e-08</td>
      <td>0.000001</td>
      <td>0.000021</td>
      <td>0.000104</td>
      <td>0.000080</td>
      <td>0.001098</td>
      <td>0.094559</td>
      <td>0.254288</td>
      <td>0.728706</td>
      <td>0.000204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.2.826.0.1.3680043.8.498.12219033294413119947...</td>
      <td>3.669435e-11</td>
      <td>2.977050e-07</td>
      <td>0.000008</td>
      <td>0.000077</td>
      <td>0.000299</td>
      <td>0.000268</td>
      <td>0.002607</td>
      <td>0.105319</td>
      <td>0.265003</td>
      <td>0.718556</td>
      <td>0.000504</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.2.826.0.1.3680043.8.498.84994474380235968109...</td>
      <td>2.364870e-03</td>
      <td>3.382650e-02</td>
      <td>0.218172</td>
      <td>0.030291</td>
      <td>0.047469</td>
      <td>0.133817</td>
      <td>0.247616</td>
      <td>0.162811</td>
      <td>0.316911</td>
      <td>0.669582</td>
      <td>0.060343</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.2.826.0.1.3680043.8.498.35798987793805669662...</td>
      <td>2.782291e-09</td>
      <td>5.089024e-06</td>
      <td>0.000089</td>
      <td>0.000344</td>
      <td>0.001175</td>
      <td>0.001187</td>
      <td>0.008067</td>
      <td>0.124998</td>
      <td>0.280071</td>
      <td>0.702830</td>
      <td>0.001846</td>
    </tr>
  </tbody>
</table>
</div>



```python
print("time: ", time.time() - start)
```

    time:  6239.787000417709

