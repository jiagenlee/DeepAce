# DeepAce

## Introduction

DeepAce is a tool for predicting lysine acetylation sites which belong of PTM questions. In this tool, I mainly use deep-learning architecture including CNN and DNN. Because of the strong ability of detecting figure-shape feature of CNN, we build it for the feature of One-Hot-Encoding of raw protein sequence. And we use DNN to detect information of physico-chemical properties.

**OneofkeyCNN.py** is a single network specialised for raw protein sequence. You can use **PreOneofkey.py** as a feature preprocessing tool, which will give you a One-Hot-Encoding of raw protein sequence.

**AAIndexDNN.py** is a network for physico-chemical properties. I select 142 properties from AAIndex database. You can generate it by **InitAAindex.py**.

**MultipleNN.py** is used to build the final network, combines above two networks and achieves the best performance.

I think the easiest way to use it is to use the source codes that include the whole network and hyperparameters.

## Build environment

Python: 3.5

Keras: 2.1.1

Keras-backend: Theano 0.9.0

